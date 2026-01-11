import os
import sys
import logging
import numpy as np
import threading
import time
from typing import List, Optional
import requests
from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType, Collection, utility
)
import vertexai
from vertexai.vision_models import Image, MultiModalEmbeddingModel

from services.product_query import fetch_products, fetch_product_by_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

# module-level cache + lock (放在檔案頂端)
_MM_MODEL = None
_MM_MODEL_LOCK = threading.Lock()

MILVUS_HOST = '127.0.0.1'
MILVUS_PORT = 19530
COLLECTION_NAME = 'uat_product_embeddings'

# Vertex location / optional project
VERTEX_LOCATION = "us-central1"
os.environ["VERTEX_PROJECT"] = "myvertexsearch"  # 可直接在此設定
VERTEX_PROJECT = os.getenv("VERTEX_PROJECT")  # 可設環境變數

# 指定 service account JSON（先設定環境變數再匯入 vertexai）
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_JSON") or os.path.join(
    os.path.dirname(__file__), "google_credential_thunderstormwang.json"
)
if os.path.exists(SERVICE_ACCOUNT_FILE):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_FILE
else:
    print(f"service account JSON not found at: `{SERVICE_ACCOUNT_FILE}`", file=sys.stderr)

logger.info(f"Using service account file: {SERVICE_ACCOUNT_FILE}")

# 初始化 Vertex（若需要 project 可設定 VERTEX_PROJECT）
if VERTEX_PROJECT:
    vertexai.init(project=VERTEX_PROJECT, location=VERTEX_LOCATION)
else:
    vertexai.init(location=VERTEX_LOCATION)


def download_image(url: str, timeout: int = 10) -> Optional[bytes]:
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        logger.warning("download_image failed for %s: %s", url, e)
        return None


def combine_img_text(img_emb: np.ndarray, txt_emb: np.ndarray) -> np.ndarray:
    """合併並正規化 image + text 向量"""
    comb = img_emb + txt_emb
    norm = np.linalg.norm(comb)
    if norm > 0:
        comb = comb / norm
    return comb.astype("float32")


def get_cached_multimodal_model(model_name: str = "multimodalembedding@001") -> MultiModalEmbeddingModel:
    """
    回傳已快取的模型；若尚未載入則在 lock 下載入一次並記錄耗時。
    可在多個函式共用，避免每次呼叫都 from_pretrained().
    """
    global _MM_MODEL
    if _MM_MODEL is not None:
        return _MM_MODEL

    with _MM_MODEL_LOCK:
        if _MM_MODEL is not None:
            return _MM_MODEL
        start = time.perf_counter()
        _MM_MODEL = MultiModalEmbeddingModel.from_pretrained(model_name)
        elapsed = time.perf_counter() - start
        logger.info(f"Loaded multimodal model {model_name} in {elapsed:.3f} s")
        return _MM_MODEL


def embed_text_and_image(alias_name: str, image_bytes: bytes) -> List[float]:
    """
    使用 Vertex 多模態模型取得 image + text 向量、合併並正規化後回傳 list[float]。
    需要在程式其他地方已經呼叫過 vertexai.init(...)。
    """
    import tempfile

    # 將 bytes 寫入暫存檔，因 Vertex SDK 目前以檔案讀取為主
    tmp_path = None
    try:
        img_temp_write_start = time.perf_counter()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tf:
            tf.write(image_bytes)
            tmp_path = tf.name
        img_temp_write_elapsed = time.perf_counter() - img_temp_write_start

        model = get_cached_multimodal_model()

        img_load_start = time.perf_counter()
        image = Image.load_from_file(tmp_path)
        img_load_elapsed = time.perf_counter() - img_load_start

        get_embeddings_start = time.perf_counter()
        embeddings = model.get_embeddings(image=image, contextual_text=alias_name)
        get_embeddings_elapsed = time.perf_counter() - get_embeddings_start

        logger.info(
            f"img_temp_write={img_temp_write_elapsed:.3f}s, img_load={img_load_elapsed:.3f}s, get_embeddings={get_embeddings_elapsed:.3f}s")

        img_emb = np.array(embeddings.image_embedding, dtype="float32")
        txt_emb = np.array(embeddings.text_embedding, dtype="float32")

        comb = combine_img_text(img_emb, txt_emb)
        # logger.info(f"ndim: {comb.ndim}, shape: {comb.shape}, size: {comb.size}, type: {comb.dtype}")

        return comb.astype("float32").tolist()
    except Exception as e:
        raise RuntimeError(f"Vertex embedding failed: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


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
    connect_milvus(MILVUS_HOST, MILVUS_PORT)

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


def print_collection_stats():
    coll = Collection(COLLECTION_NAME)
    logger.info("schema fields: %s", [(f.name, f.dtype) for f in coll.schema.fields])
    logger.info(f"num_entities: {coll.num_entities}")


def process_and_store_all(batch_size: int = 64):
    process_and_store_all_t0 = time.perf_counter()
    products = fetch_products()
    logger.info(f"fetched {len(products)} products")

    embed_dim = 1408  # 預設向量維度需與模型相符
    coll = init_milvus_collection(COLLECTION_NAME, embed_dim, MILVUS_HOST, MILVUS_PORT)

    to_insert = {
        "product_id": [],
        "sale_code": [],
        "alias_name": [],
        "embedding": []
    }

    for prod in products:
        img = None
        if prod.pic:
            pc_start = time.perf_counter()
            img = download_image(prod.pic)
            pc_end = time.perf_counter()
            logger.info(f"{download_image.__name__} took {pc_end - pc_start:.2f} seconds for productId: {prod.product_id}")
        if not img:
            logger.warning("skip product %s due to missing image", prod.product_id)
            continue

        try:
            pc_start = time.perf_counter()
            vec = embed_text_and_image(prod.alias_name, img)
            pc_end = time.perf_counter()
            logger.info(f"{embed_text_and_image.__name__} took {pc_end - pc_start:.2f} seconds for productId: {prod.product_id}")
        except Exception as e:
            logger.warning("embedding failed for %s: %s", prod.product_id, e)
            continue

        # 檢查向量維度
        if len(vec) != embed_dim:
            logger.warning("vector dim mismatch for %s: got %d expected %d", prod.product_id, len(vec), embed_dim)
            continue

        to_insert["product_id"].append(prod.product_id)
        to_insert["sale_code"].append(prod.sale_code)
        to_insert["alias_name"].append(prod.alias_name)
        to_insert["embedding"].append(vec)

        # 批次寫入
        if len(to_insert["product_id"]) >= batch_size:
            pc_start = time.perf_counter()
            coll.insert([
                to_insert["product_id"],
                to_insert["sale_code"],
                to_insert["alias_name"],
                to_insert["embedding"],
            ])
            coll.flush()
            pc_end = time.perf_counter()
            logger.info("inserted batch of %d, took %.2f seconds", len(to_insert["product_id"]), pc_end - pc_start)
            to_insert = {k: [] for k in to_insert}

    # insert remaining
    if to_insert["product_id"]:
        pc_start = time.perf_counter()
        coll.insert([
            to_insert["product_id"],
            to_insert["sale_code"],
            to_insert["alias_name"],
            to_insert["embedding"],
        ])
        coll.flush()
        pc_end = time.perf_counter()
        logger.info("inserted final batch of %d, took %.2f seconds", len(to_insert["product_id"]), pc_end - pc_start)

    logger.info(f"{__name__} completed in {time.perf_counter() - process_and_store_all_t0:.2f} seconds")
    print_collection_stats()


def search_milvus(product_id: int, top_k: int = 5):
    process_and_store_all_t0 = time.perf_counter()
    prod = fetch_product_by_id(product_id)
    if not prod:
        raise RuntimeError(f"Product with id {product_id} not found")

    logger.info(f'searching Milvus for product_id={prod.product_id}, text="{prod.alias_name}", image_path={prod.pic}')

    connect_milvus(MILVUS_HOST, MILVUS_PORT)
    try:
        coll = Collection(COLLECTION_NAME)
        coll.load()
    except Exception as e:
        connections.disconnect("default")
        raise RuntimeError(f"Failed to open/load collection `{COLLECTION_NAME}`: {e}")

    try:
        img = download_image(prod.pic)
        if not img:
            logger.warning(f"download_image returned None for {prod.pic}")
            return []

        q_vec = embed_text_and_image(prod.alias_name, img)
        logger.info(f'len: {len(q_vec)}, {q_vec}')

        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        results = coll.search([q_vec], "embedding", search_params, limit=top_k, output_fields=["product_id", "sale_code"])
        if not results or not results[0]:
            logger.info("No hits returned. Check index, nprobe, metric, and data insertion/flush.")
        else:
            logger.info("Search returned %d hits", len(results[0]))
            for hit in results[0]:
                hit_product_id = hit.entity.get("product_id")
                hit_sale_code = hit.entity.get("sale_code")
                extra = "，找到原圖" if prod.sale_code == hit_sale_code else ""
                logger.info(f"distance={hit.score:.4f}, product_id={hit_product_id}, sale_code={hit_sale_code}{extra}")
        return results
    finally:
        try:
            connections.disconnect("default")
        except Exception:
            pass


if __name__ == "__main__":
    process_and_store_all()
