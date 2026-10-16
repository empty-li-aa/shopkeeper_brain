"""
文档切分节点（上）

按 Markdown 标题切分文档，本阶段实现参数校验和标题一级切分
"""
import os,json
import re
from typing import Tuple, List
from knowledge.processor.import_processor.base import BaseNode, setup_logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.processor.import_processor.config import get_config
from knowledge.utils.markdown_util import MarkdownTableLinearizer


class DocumentSplitNode(BaseNode):
    """
    文档切分节点

    处理流程（本章部分）：
    1. 读取 MD 内容，校验参数
    2. 按 Markdown 标题进行一级切分（title 与 body 分离存储）
    """

    name = "document_split_node"

    # ------------------------------------------------------------------ #
    #                           主流程                                     #
    # ------------------------------------------------------------------ #

    def process(self, state: ImportGraphState) -> ImportGraphState:

        config = get_config()

        # 1. 获取参数
        md_content, file_title, max_content_length, min_content_length = self._validate_state(state, config)

        # 2. 根据标题切割(核心)
        sections = self._split_by_headings(md_content, file_title)

        # 3.二次切分或者合并
        final_sections = self._split_and_merge(sections, config.max_content_length, config.min_content_length)

        # 4.组装成chunk对象
        chunks = self._assemble_chunk(final_sections)

        # 5.备份(观察)
        self._backup_chunks(state, chunks)

        # 6.更新状态
        state["chunks"] = chunks

        # 7. 返回
        return state

    # ------------------------------------------------------------------ #
    #                       Step 1: 参数校验与获取                          #
    # ------------------------------------------------------------------ #

    def _validate_state(self, state: ImportGraphState, config) -> Tuple[str, str, int, int]:
        """获取输入参数并预处理"""
        self.log_step("step1", "切分文档的参数校验以及获取...")

        # 1. 获取md_content
        md_content = state.get('md_content')

        # 2. 统一换行符
        if md_content:
            md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")

        # 3. 获取文件标题
        file_title = state.get('file_title')

        # 4. 校验最大最小值
        if config.max_content_length <= 0 or config.min_content_length <= 0 \
           or config.max_content_length <= config.min_content_length:
            raise ValueError(f"切片长度参数校验失败")

        return md_content, file_title, config.max_content_length, config.min_content_length

    # ------------------------------------------------------------------ #
    #                  Step 2: 按标题一级切分                               #
    # ------------------------------------------------------------------ #

    def _split_by_headings(self, md_content: str, file_title: str) -> List[dict]:
        """
        根据 MD 的标题（1-6）级标题进行切分

        Args:
            md_content: md内容
            file_title: 文档名字

        Returns:
            sections: 列表，每个元素格式：
            {
                "title": "# 第一章",
                "body": "正文内容...",
                "file_title": "万用表",
                "parent_title": "# 第一章"
            }
        """
        self.log_step("step2", "根据标题进行切分...")

        # 1. 定义变量
        in_fence = False
        body_lines = []
        sections = []
        current_level = 0
        current_title = ""  # 当前标题
        hierarchy = [""] * 7  # 7个长度，第一个（0）不用

        # 2. 定义正则表达式
        heading_re = re.compile(r"^\s*(#{1,6})\s+(.+)")

        # 3. 切分
        content_lines = md_content.split("\n")

        def _flush():
            """封装 section 对象"""
            body = "\n".join(body_lines)
            if current_title or body:
                parent_title = ""
                for i in range(current_level - 1, 0, -1):
                    if hierarchy[i]:
                        parent_title = hierarchy[i]
                        break

                if not parent_title:
                    parent_title = current_title if current_title else file_title

                return sections.append({
                    "title": current_title if current_title else file_title,
                    "body": body,
                    "file_title": file_title,
                    "parent_title": parent_title
                })

        for content_line in content_lines:
            # 判断是否存在代码块围栏
            if content_line.strip().startswith("```") or content_line.strip().startswith("~~~"):
                in_fence = not in_fence

            match = heading_re.match(content_line) if not in_fence else None

            if match:
                _flush()

                level = len(match.group(1))  # 当前标题的级别
                current_level = level
                current_title = content_line
                hierarchy[level] = current_title

                # 清空下级标题
                for i in range(level + 1, 7):
                    hierarchy[i] = ""

                body_lines = []
            else:
                # 除了标题行以外全都收集起来
                body_lines.append(content_line)

        _flush()

        return sections


    def _split_and_merge(self, sections, max_content_length, min_content_length):
        """
        对文档进行切分和合并
        Args:
            self: self
            sections: 切表
            max_content_length: 最大内容长度
            min_content_length: 最小内容长度

        Returns:
            final_sections: 合并后的切表

        """
        # 1. 先切长的
        current_sections = []
        for section in sections:
            current_sections.extend(self._split_long_section(section, max_content_length))

        # 2. 再合短的
        final_sections = self._merge_short_section(current_sections, min_content_length)

        return final_sections

    def _split_long_section(self, section, max_content_length):
        """
        对长的 section 进行切分
        Args:
            self: self
            section: section 对象
            max_content_length: 最大内容长度

        Returns:
            sub_sections: 切分后的子 section 列表

        """
        # 1. 获取属性
        title = section.get('title')
        body = section.get('body')
        file_title = section.get('file_title')
        parent_title = section.get('parent_title')

        # # 2. 表格降维处理
        if "<table>" in body:
            self.logger.info(f"检测到表格")
            body = MarkdownTableLinearizer.process(body)

        # 3. 标题过长截断
        if len(title) > 50:
            title = title[:50]

        # 4. 计算总长度
        title_prefix = f"{title}\n\n"
        total_length = len(title_prefix) + len(body)

        # 5. 不超过阈值，无需切分
        if total_length <= max_content_length:
            return [section]

        # 6. 计算 body 可用空间
        body_length = max_content_length - len(title_prefix)
        if body_length <= 0:
            return [section]  # 防御性编程：title 本身就超长的极端情况

        # 7. 递归切分
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=body_length,
            chunk_overlap=0,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""],
            keep_separator=True
        )
        texts = text_splitter.split_text(body)

        if len(texts) == 1:
            return [section]

        # 8. 生成子 section
        sub_sections = []
        for index, text in enumerate(texts):
            sub_sections.append({
                "title": f"{title}-{index + 1}",
                "body": text,
                "file_title": file_title,
                "parent_title": parent_title,
            })
        return sub_sections

    def _merge_short_section(self, current_sections, min_content_length):
        """
        贪心累加算法：将同源的短 section 合并
        Args:
            current_sections: 切表
            min_content_length: 最小内容长度

        Returns:
            final_sections: 合并后的切表
        """
        # 1. 初始化
        current_section = current_sections[0]
        final_sections = []

        # 2. 遍历合并
        for next_section in current_sections[1:]:
            same_parent = (current_section['parent_title'] == next_section['parent_title'])

            if same_parent and len(current_section.get('body')) < min_content_length:
                # 合并 body
                current_section['body'] = (
                        current_section.get('body').rstrip() + "\n\n" + next_section.get('body').lstrip()
                )
                # 标题回退为父标题
                current_section['title'] = current_section['parent_title']
            else:
                # 封箱
                final_sections.append(current_section)
                current_section = next_section

        # 最后一个封箱
        final_sections.append(current_section)

        return final_sections

    def _assemble_chunk(self, final_chunks):
        """
        最终组合 chunk
        Args:
            final_chunks: 切表

        Returns:
            chunks: 最终组合后的 chunk 列表

        """
        chunks = []
        for chunk in final_chunks:
            title = chunk.get('title')
            file_title = chunk.get('file_title')
            parent_title = chunk.get('parent_title')
            body = chunk.get('body')
            content = f"{title}\n\n{body}"  # 标题注入到内容中

            chunks.append({
                "title": title,  # 用于检索结果展示
                "file_title": file_title,  # 用于溯源：来自哪个文档
                "parent_title": parent_title,  # 用于溯源：属于哪个章节
                "content": content,  # 用于向量嵌入和 LLM 上下文
            })
        return chunks

    def _backup_chunks(self, state, sections):
        """
        将切分结果备份到 JSON 文件
        Args:
            state: 该节点接收到的状态
            sections: 切表

        Returns:

        """
        local_dir = state.get("file_dir", "")
        if not local_dir:
            return

        try:
            os.makedirs(local_dir, exist_ok=True)
            output_path = os.path.join(local_dir, "chunks.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(sections, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"备份失败: {e}")


if __name__ == '__main__':
    setup_logging()

    document_node = DocumentSplitNode()
    # 构造状态字典
    file_path = r"D:\cx\python\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\万用表RS-12的使用\markdown_new.md"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    state = {
        "file_title": "万用表的使用",
        "md_content": content,
        "file_dir": r"D:\cx\python\shopkeeper_brain\knowledge\processor\import_processor\temp_dir"
    }
    document_node.process(state)