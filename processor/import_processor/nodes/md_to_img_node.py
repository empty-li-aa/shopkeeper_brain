import re
import time
import base64
from collections import deque
from pathlib import Path
from typing import Tuple, List, Dict, Deque, Set
from dataclasses import dataclass

from openai import OpenAI

from knowledge.utils.client.ai_clients import AIClients
from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.processor.import_processor.config import get_config
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.processor.import_processor.exceptions import (
    StateFieldError, FileProcessingError, ImageProcessingError,
)
# ── 数据模型 ──

@dataclass
class ImageContext:
    """图片在 MD 中的上下文信息。"""
    heading: str      # 最近的章节标题
    pre_text: str     # 图片上方的正文内容
    post_text: str    # 图片下方的正文内容


@dataclass
class ImageInfo:
    """一张图片的完整信息。"""
    name: str                # 图片文件名，如 "abc123.jpg"
    path: str                # 图片完整路径
    context: ImageContext    # 在 MD 中的上下文


# ── 1. 文件读写 & 备份 ──

class _MdFileHandler:
    """负责 MD 文件的读取、路径校验、图片目录构建以及处理后备份。"""
    def __init__(self, logger):
        self.logger = logger

    def read_md(self, state: ImportGraphState,node_name:str) -> Tuple[str, Path, Path]:
        """
        读取 Markdown 文件内容、路径、图片目录

        Args:
            state: 导入状态图

        Returns:
            (md_content, md_path_obj, image_dir) 元组
        """
        self.logger.info("【step_1】读取MD内容及构建图片目录")
        # 1. 获取并校验md_path是否有效
        md_path = state.get("md_path", "")
        if not md_path:
            raise StateFieldError(node_name=node_name,field_name="md_path",expected_type=str)

        # 2. 校验md_path_obj为一个真实的文件路径
        md_path_obj = Path(md_path)
        if not md_path_obj.exists():
            raise FileProcessingError(
                f"md文件路径无效: {md_path}", node_name=node_name
            )

        # 3. 读取MD文件内容
        with open(md_path_obj, "r", encoding="utf-8") as f:
            md_content = f.read()

        image_dir = md_path_obj.parent / "images"

        return md_content, md_path_obj, image_dir

    def backup(self, md_path_obj: Path, new_md_content: str) -> str:
        self.logger.info("【step_5】备份新文件")

        new_file_path = md_path_obj.with_name(
            f"{md_path_obj.stem}_new{md_path_obj.suffix}"
        )
        try:
            with open(new_file_path, "w", encoding="utf-8") as f:
                f.write(new_md_content)
            self.logger.info(f"处理后的文件已备份至: {new_file_path}")
        except IOError as e:
            self.logger.error(f"写入新文件失败 {new_file_path}: {e}")
            raise ImageProcessingError(
                f"文件写入失败: {e}", node_name="md_img_node"
            )
        return str(new_file_path)


# ── 2. 图片扫描 & 上下文提取 ──

class _ImageScanner:
    """扫描图片目录，提取每张图片在 MD 中的上下文信息。"""

    def __init__(self, logger):
        self.logger = logger

    def scan_img_dir(self,image_dir: Path,md_content: str,image_extensions: Set[str],context_length: int,) -> List[ImageInfo]:
        """
        扫描图片目录，提取每张图片在 MD 中的上下文信息。

        Args:
            image_dir: 图片目录路径
            md_content: Markdown 文件内容
            image_extensions: 图片扩展名集合
            context_length: 上下文长度

        Returns:
            图片信息列表

        """
        self.logger.info(f"【step_2】扫描图片目录 {image_dir}")

        image_list: List[ImageInfo] = []

        for img_path in Path(image_dir).iterdir():
            if not img_path.is_file():
                continue
            if img_path.suffix not in image_extensions:
                continue

            ctx = self._find_context(
                md_content, img_path.name, context_length
            )
            if ctx is None:
                self.logger.warning(
                    f"MD文件中未找到图片 {img_path.name} 的引用"
                )
                continue

            image_list.append(ImageInfo(
                name=img_path.name,
                path=str(img_path),
                context=ctx,
            ))

        self.logger.info(f"找到 {len(image_list)} 张有效图片")
        return image_list

    def _find_context(self, md_content: str, img_name: str, max_chars: int = 200) -> ImageContext | None:
        """
        返回图片在 MD 中第一次出现位置的上下文，找不到返回 None。

        Args:
            md_content: Markdown 文件内容
            img_name: 图片文件名
            max_chars: 上下文长度

        Returns:
            图片上下文信息
        """
        # 1. 编译正则表达式，匹配图片引用
        pattern = re.compile(
            r"!\[.*?\]\(.*?" + re.escape(img_name) + r".*?\)"
        )
        md_lines = md_content.split("\n")
        # 2. 遍历所有行，查找图片引用
        for line_idx, line in enumerate(md_lines):
            if not pattern.search(line):
                continue

            # 向上：找最近标题，取标题到图片之间的内容作为上文
            prev_title, prev_boundary = self._find_heading_above(
                md_lines, line_idx
            )
            pre_content = md_lines[prev_boundary + 1: line_idx]
            img_pre = self._extract_limited_context(
                pre_content, max_chars, direction="front"
            )

            # 向下：找下一个标题，取图片到标题之间的内容作为下文
            next_boundary = self._find_heading_below(md_lines, line_idx)
            post_content = md_lines[line_idx + 1: next_boundary]
            img_post = self._extract_limited_context(
                post_content, max_chars, direction="end"
            )

            return ImageContext(
                heading=prev_title,
                pre_text=img_pre,
                post_text=img_post,
            )

        return None

    @staticmethod
    def _find_heading_above(md_lines: List[str], from_idx: int) -> Tuple[str, int]:
        """
        从 from_idx 向上查找最近的标题。

        Args:
            md_lines: Markdown 文件内容行列表
            from_idx: 起始行索引

        Returns:
            标题内容和标题所在行索引
        """
        for i in range(from_idx - 1, -1, -1):
            if re.match(r"^#{1,6}\s+", md_lines[i]):
                return md_lines[i], i
        return "", -1

    @staticmethod
    def _find_heading_below(md_lines: List[str], from_idx: int) -> int:
        """
        从 from_idx 向下查找下一个标题。

        Args:
            md_lines: Markdown 文件内容行列表
            from_idx: 起始行索引

        Returns:
            下一个标题所在行索引
        """
        for i in range(from_idx + 1, len(md_lines)):
            if re.match(r"^#{1,6}\s+", md_lines[i]):
                return i
        return len(md_lines)

    @staticmethod
    def _extract_limited_context(lines: List[str], max_chars: int, direction: str) -> str:
        """
        按段落分割，按 direction 方向贪心装填，保持段落完整性。

        Args:
            lines: Markdown 文件内容行列表
            max_chars: 上下文长度
            direction: 方向，"front" 或 "end"

        Returns:
            上下文内容
        """

        current_paragraph: List[str] = [] # 临时段落
        paragraphs: List[str] = [] # 段落列表

        for line in lines:
            # 标记当前行是否为空行或其他图片引用
            is_blank_line = not line.strip()
            is_other_image = re.match(
                r"^!\[.*?\]\(.*?\)$", line.strip()
            )
            # 遇到空行或其他图片引用，结束当前段落，将当前段落添加到段落列表中
            if is_blank_line or is_other_image:
                if current_paragraph:
                    paragraphs.append("\n".join(current_paragraph))
                    current_paragraph = []
                continue

            current_paragraph.append(line)

        # 如果最后一个段落不为空，添加到段落列表中
        if current_paragraph:
            paragraphs.append("\n".join(current_paragraph))

        # 如果方向是 front，反转段落列表
        if direction == "front":
            paragraphs.reverse()

        # 贪心装填上下文，保持段落完整性
        total = 0
        selected: List[str] = []
        for para in paragraphs:
            if total + len(para) > max_chars and selected:
                break
            selected.append(para)
            total += len(para)

        # 转回来
        if direction == "front":
            selected.reverse()

        return "\n\n".join(selected)

# ── 3. VLM 摘要生成 ──

class _VLMSummarizer:
    """通过视觉语言模型为每张图片生成中文标题/摘要。"""

    def __init__(self, logger):
        self.logger = logger

    def summarize_all(self,document_title: str,image_list: List[ImageInfo],vl_model: str,requests_per_minute: int,) -> Dict[str, str]:
        """
        为文档中的所有图片生成摘要。

        Args:
            document_title: 文档标题
            image_list: 图片列表
            vl_model: 模型
            requests_per_minute: 请求频率

        Returns:
            图片摘要字典，键为图片名称，值为摘要
        """
        self.logger.info("【step_3】提取图片摘要")

        summaries: Dict[str, str] = {}
        request_timestamps: Deque[float] = deque()  # 请求时间戳队列

        try:
            client = AIClients.get_openai()
        except Exception as e:
            self.logger.warning(
                f"VLM 不可用，跳过图片摘要生成: {e}"
            )
            for img in image_list:
                summaries[img.name] = "图片描述"
            return summaries

        for img in image_list:
            self._enforce_rate_limit(request_timestamps, requests_per_minute)
            summaries[img.name] = self._summarize_one(client, vl_model, document_title, img)

        self.logger.info(f"生成 {len(summaries)} 张图片摘要")

        return summaries

    def _summarize_one(self, client: OpenAI, vl_model: str,document_title: str, img: ImageInfo,) -> str:
        """
        为单张图片生成摘要。

        Args:
            client: OpenAI 客户端
            vl_model: 模型
            document_title: 文档标题
            img: 图片信息

        Returns:
            图片摘要
        """
        parts = [p for p in (
            img.context.heading,
            img.context.pre_text,
            img.context.post_text
        ) if p]
        final_context = "\n".join(parts) if parts else "暂无可用上下文"

        try:
            with open(img.path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return "暂无图片"

        try:
            resp = client.chat.completions.create(
                model=vl_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"任务：为Markdown文档中的图片生成一个简短的中文标题。\n"
                                f"背景信息：\n"
                                f"  1. 所属文档标题：\"{document_title}\"\n"
                                f"  2. 图片上下文：{final_context}\n"
                                f"请结合图片内容和上述上下文信息，"
                                f"用中文简要总结这张图片的内容，"
                                f"生成一个精准的中文标题（不要包含图片二字）。"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}"
                            },
                        },
                    ],
                }],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            self.logger.warning(f"图片摘要生成失败 {img.path}: {e}")
            return "图片描述"

    def _enforce_rate_limit(self, timestamps: Deque[float],max_requests: int, window: int = 60,):
        """
        确保请求速率不超过指定窗口内的最大请求数。

        Args:
            timestamps: 请求时间戳队列
            max_requests: 最大请求数
            window: 时间窗口（秒）

        Returns:

        """
        now = time.time()
        while timestamps and now - timestamps[0] >= window:
            timestamps.popleft()

        if len(timestamps) >= max_requests:
            sleep_dur = window - (now - timestamps[0])
            if sleep_dur > 0:
                self.logger.info(
                    f"达到速率限制，暂停 {sleep_dur:.2f} 秒..."
                )
                time.sleep(sleep_dur)
            now = time.time()
            while timestamps and now - timestamps[0] >= window:
                timestamps.popleft()

        timestamps.append(now)


# ── 4. MinIO 上传 & MD 内容替换 ──

class _ImageUploader:
    """将本地图片上传至 MinIO，并在 MD 内容中替换为远程 URL + 摘要。"""

    def __init__(self, logger):
        self.logger = logger

    def upload_and_replace(self,
                           document_name: str, md_content: str,images_summaries: Dict[str, str],
                           image_list: List[ImageInfo],minio_bucket: str, minio_base_url: str) -> str:
        """
        上传图片到 MinIO 并更新 MD 内容。

        Args:
            document_name: 文档名称
            md_content: Markdown 内容
            images_summaries: 图片摘要映射
            image_list: 图片列表
            minio_bucket: MinIO 桶名
            minio_base_url: MinIO 基础 URL

        Returns:
            更新后的 Markdown 内容

        """

        self.logger.info("【step_4】上传图片到MinIO并更新MD")

        remote_urls = self._upload_all(
            document_name, image_list, minio_bucket, minio_base_url
        )
        return self._replace_in_md(
            md_content, images_summaries, remote_urls
        )

    def _upload_all(
            self, document_name: str, image_list: List[ImageInfo],
            minio_bucket: str, minio_base_url: str,
    ) -> Dict[str, str]:
        remote_urls: Dict[str, str] = {}

        try:
            minio_client = StorageClients.get_minio()
        except Exception as e:
            self.logger.warning(
                f"MinIO 不可用，所有图片保留本地路径: {e}"
            )
            for img in image_list:
                remote_urls[img.name] = img.path
            return remote_urls

        for img in image_list:
            object_name = f"{document_name}/{img.name}"
            try:
                minio_client.fput_object(
                    minio_bucket, object_name, img.path
                )
                remote_url = (
                    f"{minio_base_url}/{minio_bucket}/{object_name}"
                )
                self.logger.info(f"{img.name} 上传成功")
                remote_urls[img.name] = remote_url
            except Exception:
                self.logger.warning(
                    f"{img.name} 上传失败，保留本地路径"
                )
                remote_urls[img.name] = img.path

        self.logger.info(
            f"成功上传 {len(remote_urls)} 张图片到 MinIO"
        )
        return remote_urls

    @staticmethod
    def _replace_in_md(md_content: str,summaries: Dict[str, str],remote_urls: Dict[str, str],) -> str:
        """
        替换 MD 中的图片引用为远程 URL + 摘要。

        Args:
            md_content: Markdown 内容
            summaries: 图片摘要映射
            remote_urls: 图片远程 URL 映射

        Returns:
            更新后的 Markdown 内容

        """
        pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")

        def replacer(match: re.Match) -> str:
            original_path = match.group(2).strip()
            file_name_in_md = Path(original_path).name
            for img_name, summary in summaries.items():
                if img_name == file_name_in_md:
                    return f"![{summary}]({remote_urls[img_name]})"
            return match.group(0)

        return pattern.sub(replacer, md_content)

# ── 主节点 ──

class _MarkDownImageNode(BaseNode):
    """处理 MarkDown 图片的管道节点 —— 仅负责编排，不含业务细节。"""

    name = "md_img_node"

    def __init__(self):
        super().__init__()
        self.file_handler = _MdFileHandler(self.logger)
        self.scanner = _ImageScanner(self.logger)
        self.summarizer = _VLMSummarizer(self.logger)
        self.uploader = _ImageUploader(self.logger)

    def process(self, state: ImportGraphState) -> ImportGraphState:

        config = get_config()

        # 1. 读取文件
        md_content, md_path_obj, image_dir = self.file_handler.read_md(state,self.name)

        if not image_dir.exists():
            self.logger.warning(f"文件 {md_path_obj.name} 暂无图片要处理")
            state["md_content"] = md_content
            return state

        # 2. 扫描图片 & 提取上下文
        image_list = self.scanner.scan_img_dir(
            image_dir, md_content,
            image_extensions=config.image_extensions,
            context_length=config.img_content_length,
        )

        # 3. VLM 生成摘要
        summaries = self.summarizer.summarize_all(
            document_title=md_path_obj.stem, # 文档标题
            image_list=image_list,           # 图片列表
            vl_model=config.vl_model,        # 模型
            requests_per_minute=config.requests_per_minute, # 请求频率
        )
        # 4. 上传 & 替换
        new_md_content = self.uploader.upload_and_replace(
            document_name=md_path_obj.stem,               # 文档名称
            md_content=md_content,                        # Markdown 内容
            images_summaries=summaries,                   # 图片摘要映射
            image_list=image_list,                        # 图片列表
            minio_bucket=config.minio_bucket,             # MinIO 桶名
            minio_base_url=config.get_minio_base_url(),    # MinIO 基础 URL
        )
        # 5. 备份
        self.file_handler.backup(md_path_obj, new_md_content)

        # 6. 更新并返回 state
        state["md_content"] = new_md_content
        return state


if __name__ == "__main__":
    setup_logging()

    node = _MarkDownImageNode()
    state = {
        "md_path": r"D:\cx\python\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\万用表RS-12的使用\markdown.md"
    }
    node.process(state)