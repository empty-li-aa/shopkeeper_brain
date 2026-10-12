from pathlib import Path
from typing import Tuple
import os
import time
import json
import shutil
import zipfile
import subprocess

from knowledge.processor.import_processor.base import BaseNode, T, setup_logging
from knowledge.processor.import_processor.exceptions import ValidationError, FileProcessingError, PdfConversionError
from knowledge.processor.import_processor.state import ImportGraphState


# venv 里的命令行工具；mineru 包没有 __main__，不能用 python -m mineru 调用
_SCRIPTS_DIR = Path(__file__).resolve().parents[3] / ".venv" / "Scripts"
MINERU_KIT_EXE = _SCRIPTS_DIR / "mineru-kit.exe"

# 模型放在 D 盘；不显式指定时服务会去 C:\Users\<用户>\.mineru 找模型并报缺失
MINERU_HOME = r"D:\models\mineru"

# mineru-kit 输出 zip 内的固定文件名
ZIP_MD_NAME = "markdown.md"


class PdfToMdNode(BaseNode):

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        将 PDF 文件转换为 Markdown 文件

        :arg state: 导入状态图

        :return: 更新后的状态
        """

        #1.对参数校验
        import_file_path_obj , file_dir_obj = self._validate_state(state)
        # 2. 利用MinerU工具解析pdf成为md
        processed_code = self._execute_mineru(import_file_path_obj, file_dir_obj)
        if processed_code != 0:
            raise PdfConversionError("MinerU解析PDF失败", self.name)

        # 3. 解压 zip，取出 markdown 和 images 目录
        self._extract_mineru_zip(import_file_path_obj, file_dir_obj)

        # 4. 获取md的path
        md_path = self._get_md_paths(import_file_path_obj, file_dir_obj)

        # 5. 更新state 字典的md_path
        state['md_path'] = md_path

        # 5. 返回state
        return state



    def _validate_state(self, state:ImportGraphState) -> Tuple[Path,Path]:
        """
        验证输入路径

        Args:
            state: 该节点接收到的状态

        Returns:
            (import_file_path_obj, file_dir_path_obj) 元组
        """

        self.log_step("step1","校验import_file_path是否有效")
        # 1.获取输入pdf文件路径
        import_file_path = state.get("import_file_path",'')
        #2.获取解析后的文件目录
        file_dir = state.get("file_dir",'')

        #3.校验import_file_path是否有效
        if not import_file_path:
            raise ValidationError("解析的文件不存在",self.name)
        #4.Path标准化
        import_file_path_obj = Path(import_file_path)
        # 5.校验是否为一个真实的地址
        if not import_file_path_obj.exists():
            raise FileProcessingError("解析的文件路径不存在",self.name)

        # 6.判断输出目录是否存在
        if not file_dir:
            # 默认兜底
            file_dir = import_file_path.parent

        # 7.判断输出目录是否有效
        file_dir_obj = Path(file_dir)
        if not file_dir_obj.exists():
            raise ValidationError("解析后的文件目录不存在",self.name)
        # 8.标准输出目录
        self.logger.info(f"上传文件路径: {import_file_path_obj}")
        self.logger.info(f"解析后的文件目录: {file_dir_obj}")

        return import_file_path_obj,file_dir_obj

    def _execute_mineru(self, import_file_path:Path, file_dir:Path)->int:
        """
                执行 MinerU 命令，输出带图片的 zip 包

                MinerU 4 把「解析」和「取图」拆成了两个动作：`mineru parse` 只产出
                markdown，图片位置仅写 doc: 定位符，不落地任何图片文件。而
                `mineru-kit parse -f zip` 会一次性打包 markdown + images/ + JSON。

                Args:
                    import_file_path: 解析的文件路径
                    file_dir: 解析后的文件目录

                Returns:
                    命令返回码（0 表示成功）
                """
        self.log_step("step2", "执行MinerU解析PDF")

        # 1. 构建命令行
        #    注意 -p 是 --pages（页码）不是路径，源文件必须作为位置参数传入
        cmd = [
            str(MINERU_KIT_EXE),
            "parse",
            str(import_file_path),
            "-o",
            str(file_dir),
            "-f",
            "zip",
            "-p",
            "all",
        ]

        # 2. 模型在 D 盘，必须把 MINERU_HOME 传给子进程
        env = os.environ.copy()
        env.setdefault("MINERU_HOME", MINERU_HOME)

        process_start_time = time.time()

        # 3. 执行命令行(子进程执行命令行)
        proc = subprocess.Popen(
            args=cmd,
            env=env,
            stdout=subprocess.PIPE, # 成功日志输出标准输出
            stderr=subprocess.STDOUT, # 错误日志输出标准输出
            errors="replace",  # 遇到乱码时替换？、棱形
            text=True,  # 输出的内容是字符串 不是字节
            encoding="utf-8",  # 用指定的中文字符集进行编解码
            bufsize=1  # 按行缓冲，只要缓冲区一行满了就输出
        )

        # 4. 获取日志信息
        for line in proc.stdout:
            self.logger.info(f"执行MinerU产生的日志：{line.rstrip()}")

        # 5. 等待子进程做完
        processed_code = proc.wait()

        process_end_time = time.time()
        if processed_code == 0:
            self.logger.info(
                f"MinerU成功解析PDF文件：{import_file_path.name} "
                f"耗时:{process_end_time - process_start_time:.2f}s"
            )
        else:
            self.logger.error(f"MinerU解析PDF文件：{import_file_path.name}失败")

        # 6. 返回状态码
        return processed_code

    def _extract_mineru_zip(self, import_file_path:Path, file_dir:Path)->None:
        """
                解压 MinerU 产出的 zip，取出 markdown 与 images 目录

                zip 结构:
                  <文件名>.zip
                    ├── markdown.md              # 图片引用为 images/xxx.jpg 相对路径
                    ├── images/
                    ├── middle_json.json
                    ├── model_output.json
                    └── structured_content.json

                解压后目录:
                  file_dir/
                    └── 文件名/
                         ├── markdown.md
                         └── images/

                Args:
                    import_file_path: PDF 文件 Path 对象
                    file_dir: 输出目录 Path 对象

                Returns:
                    无
                """
        self.log_step("step3", "解压MinerU产出的zip")

        zip_path = file_dir / f"{import_file_path.stem}.zip"
        if not zip_path.exists():
            raise PdfConversionError(f"未找到MinerU输出的zip包: {zip_path}", self.name)

        extract_dir = file_dir / import_file_path.stem
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path) as zf:
            # 防 zip slip：只解压落在 extract_dir 内的条目
            for member in zf.namelist():
                member_path = (extract_dir / member).resolve()
                if not member_path.is_relative_to(extract_dir.resolve()):
                    raise PdfConversionError(f"zip包内存在非法路径: {member}", self.name)
            zf.extractall(extract_dir)

        if not Path(self._get_md_paths(import_file_path, file_dir)).exists():
            raise PdfConversionError(f"zip包内未找到 {ZIP_MD_NAME}", self.name)

        image_dir = extract_dir / "images"
        image_count = len(list(image_dir.glob("*"))) if image_dir.is_dir() else 0
        self.logger.info(f"解压完成，导出图片 {image_count} 张 -> {image_dir}")


    def _get_md_paths(self, import_file_path:Path, file_dir:Path)-> str:
        """
                获取转换结果路径

                MinerU 4 的 mineru-kit 输出 zip，解压后结构为:
                file_dir/
                  └── 文件名/
                       ├── markdown.md
                       └── images/

                Args:
                    import_file_path: PDF 文件 Path 对象
                    file_dir: 输出目录 Path 对象

                Returns:
                    Markdown 文件路径字符串
                """
        return str(file_dir / import_file_path.stem / ZIP_MD_NAME)




# ================================================================== #
#                        测试                                        #
# ================================================================== #

if __name__ == '__main__':
    setup_logging()
    pdf_to_md_node = PdfToMdNode()

    pdf_to_md_node_init_state = {
        "import_file_path": r"D:\cx\python\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\万用表RS-12的使用.pdf",
        "file_dir": r"D:\cx\python\shopkeeper_brain\knowledge\processor\import_processor\temp_dir"
    }
    processed_result = pdf_to_md_node.process(pdf_to_md_node_init_state)

    print(json.dumps(processed_result, indent=4, ensure_ascii=False))
