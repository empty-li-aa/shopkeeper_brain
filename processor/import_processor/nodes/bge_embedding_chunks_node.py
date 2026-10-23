
"""
BGE-M3 Embedding Chunks Node
负责将文档chunks进行批量向量嵌入（稠密+稀疏），并将结果注入到每个chunk中。
"""

import json
from typing import Dict, List, Any, Tuple
from pathlib import Path

from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.processor.import_processor.exceptions import ValidationError, EmbeddingError
from knowledge.processor.import_processor.exceptions import StateFieldError
from knowledge.utils.client.ai_clients import AIClients



class BgeEmbeddingChunksNode(BaseNode):
    """
    对所有 chunks 进行 BGE-M3 向量嵌入：
    1. 校验输入，获取 chunks 列表
    2. 按批次拼接 embedding_content（item_name + content），调用模型批量编码
    3. 从编码结果中提取稠密向量和稀疏向量，注入到每个 chunk 中
    """

    name = "bge_embedding_chunks_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:

        # 1. 校验state
        chunks = self._validate_state(state)

        # 2. 获取嵌入模型（仅加载一次，所有批次共享）
        try:
            embedding_model = AIClients.get_bge_m3_client()
        except Exception as e:
            self.logger.error(f"BGE-M3嵌入模型获取失败 原因:{str(e)}")
            raise EmbeddingError(message="M3嵌入模型获取失败", node_name=self.name)

        # 3. 获取要嵌入chunk的长度
        total = len(chunks)
        batch_size = self.config.embedding_batch_size

        # 4. 批量嵌入
        embed_result = []
        for batch_start in range(0, total, batch_size):
            batch = chunks[batch_start:batch_start + batch_size]
            batch_end = batch_start + len(batch)
            self.logger.info(f"嵌入批次 [{batch_start + 1}-{batch_end}] / {total}")
            result = self._embed_chunks(batch, embedding_model)
            embed_result.extend(result)

        return embed_result

    def _validate_state(self, state: ImportGraphState):
        chunks = state.get('chunks')

        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(
                node_name=self.name, field_name="chunks", expected_type=List
            )

        # 校验每个 chunk 必须是 dict
        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                raise ValidationError(
                    f"chunk[{i}] 类型无效 期望类型是 {type(chunk).__name__}", self.name
                )

        return chunks

    def _embed_chunks(self, batch: List[Dict[str, Any]], embedding_model):
        # 1. 拼接要嵌入的内容
        texts = [
            f"{chunk.get('item_name', '')}\n{chunk.get('content', '')}"
            for chunk in batch
        ]

        # 2. 批量嵌入
        try:
            vector_result = embedding_model.encode_documents(texts)
        except Exception as e:
            raise EmbeddingError(message=str(e), node_name=self.name)

        # 3. 判断向量结果
        if not vector_result:
            raise EmbeddingError(f"BGE-M3编码返回空结果", node_name=self.name)

        # 4. 将向量注入每个 chunk
        sparse_csr = vector_result['sparse']
        for i, chunk in enumerate(batch):
            chunk['dense_vector'] = vector_result.get('dense')[i].tolist()
            chunk['sparse_vector'] = self._extract_sparse_vector(sparse_csr, i)

        return batch

    def _extract_sparse_vector(self, sparse_csr, index) -> Dict[str, Any]:
        start_index = sparse_csr.indptr[index]
        end_index = sparse_csr.indptr[index + 1]

        token_id = sparse_csr.indices[start_index:end_index].tolist()
        weight = sparse_csr.data[start_index:end_index].tolist()

        return dict(zip(token_id, weight))

if __name__ == '__main__':
    setup_logging()

    base_dir = Path(
        r"D:\cx\python\shopkeeper_brain\knowledge\processor\import_processor\temp_dir"
    )
    input_path = base_dir / "chunks_item_name.json"
    output_path = base_dir / "chunks_vector.json"

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    node = BgeEmbeddingChunksNode()
    result_state = node.process({"chunks": chunks_data.get('chunks')})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_state, f, ensure_ascii=False, indent=4)

    print(f"向量生成完成，结果已保存至:\n{output_path}")