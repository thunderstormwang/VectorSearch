import logging
import time

from pymilvus import connections, Collection, FieldSchema, DataType, CollectionSchema
from pymilvus.orm import utility

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def connect_milvus(host: str, port: int, retries: int = 5, delay: float = 2.0):
    last_exc = None
    for attempt in range(retries):
        try:
            connections.connect(alias="default", host=host, port=port)
            return
        except Exception as e:
            last_exc = e
            time.sleep(delay)
    raise RuntimeError(f"Failed to connect to Milvus at {host}:{port} after {retries} retries: {last_exc}")


def init_milvus_collection(
        collection_name: str,
        dim: int,
        host: str,
        port: int,
        replicas: int = 1
) -> Collection:
    """
    連線 Milvus，若 collection 已存在則刪除再重建，回傳已建立並 load 的 Collection。
    `replicas` 會被當作 shards_num 傳給 Collection（可視需求調整）。
    """
    connect_milvus(host, port)

    # 若存在則嘗試刪除（Collection.drop() -> fallback utility.drop_collection）
    if utility.has_collection(collection_name):
        try:
            try:
                Collection(collection_name).drop()
                logger.info("Dropped existing Milvus collection %s", collection_name)
            except Exception:
                utility.drop_collection(collection_name)
                logger.info("Dropped existing Milvus collection %s via utility.drop_collection", collection_name)
        except Exception as e:
            raise RuntimeError(f"Failed to drop existing collection `{collection_name}`: {e}")

    fields = [
        FieldSchema(name="product_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="sale_code", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="alias_name", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields=fields, description="Product embeddings")
    coll = Collection(name=collection_name, schema=schema)
    logger.info("Created Milvus collection %s (dim=%d)", collection_name, dim)
    # 建議建立向量索引（依需求調整 index type / params）
    index_params = {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 128}}
    coll.create_index(field_name="embedding", index_params=index_params)
    coll.load()
    return coll


def print_collection_stats(collection_name: str):
    coll = Collection(collection_name)
    logger.info("schema fields: %s", [(f.name, f.dtype) for f in coll.schema.fields])
    logger.info(f"num_entities: {coll.num_entities}")
