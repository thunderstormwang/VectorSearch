import asyncio
import os
import sys
import logging
import numpy as np
import time
import threading
from typing import List, Optional, Dict
import vertexai
from vertexai.vision_models import Image, MultiModalEmbeddingModel

from models.product_record import ProductRecord

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

# module-level cache + lock (放在檔案頂端)
_MM_MODEL = None
_MM_MODEL_LOCK = threading.Lock()

# Vertex location / optional project
VERTEX_LOCATION = "us-central1"
os.environ["VERTEX_PROJECT"] = "myvertexsearch"  # 可直接在此設定
VERTEX_PROJECT = os.getenv("VERTEX_PROJECT")  # 可設環境變數

# 指定 service account JSON（先設定環境變數再匯入 vertexai）
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_JSON") or os.path.join(
    os.path.dirname(__file__), "../google_credential_thunderstormwang.json"
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


def combine_img_text(img_emb: np.ndarray, txt_emb: np.ndarray) -> np.ndarray:
    """合併並正規化 image + text 向量"""
    comb = img_emb + txt_emb
    norm = np.linalg.norm(comb)
    if norm > 0:
        comb = comb / norm
    return comb.astype("float32")


def embed_text_and_image(alias_name: str, image_bytes: bytes) -> List[float]:
    """
    使用 Vertex 多模態模型取得 image + text 向量、合併並正規化後回傳 list[float]。
    需要在程式其他地方已經呼叫過 vertexai.init(...)。
    """
    import tempfile

    # 將 bytes 寫入暫存檔，因 Vertex SDK 目前以檔案讀取為主
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tf:
            tf.write(image_bytes)
            tmp_path = tf.name

        model = get_cached_multimodal_model()
        image = Image.load_from_file(tmp_path)
        embeddings = model.get_embeddings(image=image, contextual_text=alias_name)

        img_emb = np.array(embeddings.image_embedding, dtype="float32")
        txt_emb = np.array(embeddings.text_embedding, dtype="float32")

        comb = combine_img_text(img_emb, txt_emb)

        return comb.astype("float32").tolist()
    except Exception as e:
        raise RuntimeError(f"Vertex embedding failed: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _embed_and_time(prod: ProductRecord, img: bytes) -> tuple[int, list[float], float]:
    t0 = time.perf_counter()
    vec = embed_text_and_image(prod.alias_name, img)
    elapsed = time.perf_counter() - t0
    # logger.info(f"get_embeddings took {elapsed:.2f} seconds for productId: {prod.product_id}")
    return prod.product_id, vec, elapsed


async def batch_get_embeddings(products: List[ProductRecord], images_by_id: Dict[int, bytes], max_workers: int = 8):
    """
    同樣用 Semaphore + asyncio.to_thread 執行同步的 _embed_and_time，確保最多 max_workers 個同時執行。
    回傳 product_id -> vector (或 None) 的映射。
    """
    sem = asyncio.Semaphore(max_workers)

    async def worker(prod):
        img = images_by_id.get(prod.product_id)
        if img is None:
            logger.warning(f"no image for productId: {prod.product_id}, skipping embedding")
            return prod.product_id, None, 0.0
        async with sem:
            return await asyncio.to_thread(_embed_and_time, prod, img)

    tasks = [asyncio.create_task(worker(p)) for p in products]
    embeddings: Dict[int, Optional[list[float]]] = {}
    for fut in asyncio.as_completed(tasks):
        product_id, vec, elapsed = await fut
        embeddings[product_id] = vec
    return embeddings