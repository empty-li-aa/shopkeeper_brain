"""
Milvus Import Node
负责将带有向量的 chunks 批量写入 Milvus，并回填 chunk_id。

架构：门面 + 建造者模式
- MilvusImportNode (门面): 校验 → 确保集合存在 → 插入 → 更新 state
- _MilvusSchemaBuilder: 构建集合 Schema（主键 + 向量 + 标量）
- _MilvusIndexBuilder: 构建集合索引
- _MilvusInserter: 执行插入并回填 chunk_id
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pymilvus import DataType, MilvusClient
from pymilvus.orm.schema import CollectionSchema

from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.processor.import_processor.exceptions import ValidationError
from knowledge.utils.client.storage_clients import StorageClients

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ScalarFieldSpec:
    field_name: str
    datatype: DataType
    max_length: Optional[int] = None


_SCALAR_FIELDS: Sequence[ScalarFieldSpec] = (
    ScalarFieldSpec(field_name="content",      datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="title",         datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="parent_title",  datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="file_title",    datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="item_name",     datatype=DataType.VARCHAR, max_length=65535),
)


class _MilvusSchemaBuilder:
    @staticmethod
    def build(client: MilvusClient, dim: int) -> CollectionSchema:
        logger.info("开始构建 Schema...")
        # 创建动态字段 Schema （后续可扩展字段）
        schema = client.create_schema(enable_dynamic_field=True)

        schema.add_field(field_name="chunk_id", datatype=DataType.INT64,
                         is_primary=True, auto_id=True)
        schema.add_field(field_name="dense_vector",
                         datatype=DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field(field_name="sparse_vector",
                         datatype=DataType.SPARSE_FLOAT_VECTOR)

        for spec in _SCALAR_FIELDS:
            kwargs: Dict[str, Any] = {
                "field_name": spec.field_name,
                "datatype": spec.datatype,
            }
            # 不一定所以字段都有最大长度
            if spec.max_length is not None:
                kwargs["max_length"] = spec.max_length
            schema.add_field(**kwargs)

        logger.info("Schema 构建完成")
        return schema


class _MilvusIndexBuilder:
    @staticmethod
    def build(client: MilvusClient, collection_name: str):
        logger.info(f"开始构建集合 {collection_name} 索引...")
        index = client.prepare_index_params(collection_name=collection_name)

        index.add_index(field_name="dense_vector",
                        index_name="dense_vector_index",
                        index_type="AUTOINDEX",
                        metric_type="COSINE")
        index.add_index(field_name="sparse_vector",
                        index_name="sparse_vector_index",
                        index_type="SPARSE_INVERTED_INDEX",
                        metric_type="IP")

        logger.info(f"集合 {collection_name} 索引构建完成")
        return index


class _MilvusInserter:
    def __init__(self, client: MilvusClient, collection_name: str):
        self._client = client
        self._collection_name = collection_name

    def insert(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info(f"开始插入 {len(chunks)} 条记录到 Milvus...")
        result = self._client.insert(
            collection_name=self._collection_name, data=chunks
        )
        ids = result.get("ids", [])
        insert_count = result.get("insert_count", 0)

        for chunk, chunk_id in zip(chunks, ids):
            chunk["chunk_id"] = chunk_id

        logger.info(f"插入完成：{insert_count} 条记录，已回填 chunk_id")
        return chunks



class ImportMilvusNode(BaseNode):
    name = "import_milvus_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:

        # 1. 校验参数
        validated_chunks, dim = self._validate_state(state)

        # 2. 确保集合存在
        milvus_client = StorageClients.get_milvus_client()
        if milvus_client is None:
            raise ValidationError("无法获取 Milvus 客户端", self.name)

        # 3. 创建集合
        collection = self.config.chunks_collection
        self._create_collection(milvus_client, collection, dim)

        # 4. 插入 chunks
        inserter = _MilvusInserter(client=milvus_client, collection_name=collection)
        state["chunks"] = inserter.insert(chunks=validated_chunks)

        # 5. 返回结果
        return state



    def _validate_state(self, state):
        self.log_step("validate", "参数校验")
        chunks = state.get("chunks")
        if not chunks or not isinstance(chunks, list):
            raise ValidationError("待入库的 chunks 为空或类型无效", self.name)

        validated = []
        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                raise ValidationError(
                    f"chunks[{i}] 类型无效：期望 dict，实际为 {type(chunk).__name__}",
                    self.name,
                )
            if chunk.get("dense_vector") and chunk.get("sparse_vector"):
                validated.append(chunk)
            else:
                self.logger.warning(f"chunks[{i}] 缺少混合向量，已跳过")

        if not validated:
            raise ValidationError("所有 chunk 均无有效向量，无法入库", self.name)

        dim = len(validated[0]["dense_vector"])
        self.logger.info(f"有效 chunks：{len(validated)}，向量维度：{dim}")
        return validated, dim

    def _create_collection(self, client, collection_name, dim):
        self.log_step("collection", f"检查集合 {collection_name}")
        if client.has_collection(collection_name=collection_name):
            self.logger.info(f"集合 {collection_name} 已存在，跳过创建")
            return

        # 创建集合
        schema = _MilvusSchemaBuilder.build(client, dim)
        # 创建索引
        index = _MilvusIndexBuilder.build(client, collection_name)
        client.create_collection(
            collection_name=collection_name, schema=schema, index_params=index
        )
        self.logger.info(f"集合 {collection_name} 创建完成")


def _cli_main() -> None:

    import  json
    setup_logging()

    temp_dir = Path(r"D:\cx\python\shopkeeper_brain\knowledge\processor\import_processor\temp_dir"
    )
    input_path = temp_dir / "chunks_vector.json"
    output_path = temp_dir / "chunks_vector_ids.json"

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        content = json.load(f)

    state: ImportGraphState = {"chunks": content}

    node = ImportMilvusNode()
    result_state = node.process(state)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_state, f, ensure_ascii=False, indent=4)

    print(f"结果已保存至: {output_path}")

if __name__ == '__main__':
    _cli_main()
