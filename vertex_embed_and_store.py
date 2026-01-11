import os
import sys
import time
import numpy as np
from typing import Optional, Tuple, List
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)
import vertexai
from vertexai.vision_models import Image, MultiModalEmbeddingModel

# --- 設定 ---
IMAGE_DIR = r'./images'
COLLECTION_NAME = 'image_clip'
MILVUS_HOST = '127.0.0.1'
MILVUS_PORT = 19530
BATCH_SIZE = 32
FIXED_TEXT = "人類法師"
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

print(f"Using service account file: {SERVICE_ACCOUNT_FILE}")

# 初始化 Vertex（若需要 project 可設定 VERTEX_PROJECT）
if VERTEX_PROJECT:
    vertexai.init(project=VERTEX_PROJECT, location=VERTEX_LOCATION)
else:
    vertexai.init(location=VERTEX_LOCATION)


# 輔助：呼叫 Vertex 多模態模型，回傳 (image_embedding, text_embedding)
def get_multi_modal_embeddings(
        image_path: str,
        contextual_text: str = FIXED_TEXT,
        location: str = VERTEX_LOCATION,
) -> Tuple[np.ndarray, np.ndarray]:
    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
    image = Image.load_from_file(image_path)
    embeddings = model.get_embeddings(image=image, contextual_text=contextual_text)
    img_emb = np.array(embeddings.image_embedding, dtype="float32")
    txt_emb = np.array(embeddings.text_embedding, dtype="float32")
    return img_emb, txt_emb


def get_text_embedding(contextual_text: str) -> np.ndarray:
    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
    embeddings = model.get_embeddings(contextual_text=contextual_text)
    txt_emb = np.array(embeddings.text_embedding, dtype="float32")
    # normalize
    norm = np.linalg.norm(txt_emb)
    if norm > 0:
        txt_emb = txt_emb / norm
    return txt_emb.astype("float32")


def combine_img_text(img_emb: np.ndarray, txt_emb: np.ndarray) -> np.ndarray:
    comb = img_emb + txt_emb
    norm = np.linalg.norm(comb)
    if norm > 0:
        comb = comb / norm
    return comb.astype("float32")


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


def prepare_milvus_collection(emb_dim: int):
    connect_milvus(MILVUS_HOST, MILVUS_PORT)

    if utility.has_collection(COLLECTION_NAME):
        try:
            Collection(COLLECTION_NAME).drop()
        except Exception:
            try:
                utility.drop_collection(COLLECTION_NAME)
            except Exception as e:
                raise RuntimeError(f"Failed to drop existing collection: {e}")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=emb_dim),
        FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=1024),
    ]
    schema = CollectionSchema(fields, description="Vertex multimodal image+text embeddings")
    coll = Collection(name=COLLECTION_NAME, schema=schema)
    return coll


def create_index_and_load(collection):
    index_params = {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 128}}
    collection.create_index(field_name="embedding", index_params=index_params)
    collection.load()


def print_collection_stats():
    coll = Collection(COLLECTION_NAME)
    print("schema fields:", [(f.name, f.dtype) for f in coll.schema.fields])
    print("num_entities:", coll.num_entities)


def vertex_main():
    """
    texts: 若提供，應為字串陣列，長度至少與 images 數量相同，texts[i] 會與 images[i] 對應使用。
    若為 None，會對每張圖片使用預設的 FIXED_TEXT。
    """
    files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not files:
        print("No images found in", IMAGE_DIR)
        return

    # 以第一張圖片檔名（去副檔名）作為 sample_text 以取得向量維度
    try:
        sample_text = os.path.splitext(files[0])[0] if files else FIXED_TEXT
        sample_img_path = os.path.join(IMAGE_DIR, files[0])
        sample_img_emb, _sample_txt_emb = get_multi_modal_embeddings(sample_img_path, sample_text)
        emb_dim = sample_img_emb.shape[0]
    except Exception as e:
        print("Failed to get sample embeddings from Vertex:", e)
        return

    try:
        collection = prepare_milvus_collection(emb_dim)
        print("Connected to Milvus and created collection:", COLLECTION_NAME)
    except Exception as e:
        print("Failed to connect/create Milvus collection:", e)
        return

    emb_batch: List[List[float]] = []
    paths_batch: List[str] = []

    for i, fname in enumerate(files):
        path = os.path.join(IMAGE_DIR, fname)
        contextual_text = os.path.splitext(fname)[0]
        try:
            img_emb, txt_emb = get_multi_modal_embeddings(path, contextual_text)
        except Exception as e:
            print("skip", path, e)
            continue

        combined = combine_img_text(img_emb, txt_emb)
        emb_batch.append(combined.tolist())
        paths_batch.append(path)

        if len(emb_batch) >= BATCH_SIZE or i == len(files) - 1:
            try:
                # 注意 order 對應 schema 中非 auto_id 欄位順序： embedding, path
                collection.insert([emb_batch, paths_batch])
                collection.flush()
                print(f"Inserted batch up to file: {fname} (count {len(emb_batch)})")
            except Exception as e:
                print("Insert failed:", e)
            emb_batch = []
            paths_batch = []

    try:
        print("Creating index and loading collection...")
        create_index_and_load(collection)
        print("Index created and collection loaded.")
    except Exception as e:
        print("Index/create/load failed:", e)

    print_collection_stats()

    try:
        connections.disconnect("default")
        print("Disconnected from Milvus.")
    except Exception:
        pass


def search_milvus(image_path: Optional[str] = None, text: Optional[str] = None, top_k: int = 5):
    """
    在 Milvus 中查詢。\n
    參數:\n
      * image_path: 查詢圖片路徑（可選）\n
      * text: 查詢文字（可選）\n
    兩者至少提供一個。若同時提供，使用該圖片與該文字產生向量；\n
    若只有圖片，會以 FIXED_TEXT 作為 contextual text；若只有文字，使用文字向量做查詢。
    """
    if not image_path and not text:
        raise ValueError("Provide at least `image_path` or `text`.")

    connect_milvus(MILVUS_HOST, MILVUS_PORT)
    try:
        coll = Collection(COLLECTION_NAME)
        coll.load()
    except Exception as e:
        connections.disconnect("default")
        raise RuntimeError(f"Failed to open/load collection `{COLLECTION_NAME}`: {e}")

    try:
        if image_path and text:
            img_emb, txt_emb = get_multi_modal_embeddings(image_path, text)
            q_vec = combine_img_text(img_emb, txt_emb).tolist()
        elif image_path:
            img_emb, txt_emb = get_multi_modal_embeddings(image_path, FIXED_TEXT)
            q_vec = combine_img_text(img_emb, txt_emb).tolist()
        else:
            txt_emb = get_text_embedding(text)
            q_vec = txt_emb.tolist()

        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        results = coll.search([q_vec], "embedding", search_params, limit=top_k, output_fields=["id", "path"])
        print("Search results:")
        for hit in results[0]:
            print(f"distance: {hit.score:.4f}, path: {hit.entity.get('path')}, id: {hit.entity.get('id')}")
        return results
    finally:
        try:
            connections.disconnect("default")
        except Exception:
            pass


def search_many_texts(texts: List[str], top_k: int = 5):
    """
    使用字串陣列逐一在 Milvus 查詢並印出結果距離與 path。
    參數:
      * texts: 字串陣列，逐一當作查詢文字
      * top_k: 每個查詢回傳的 top k
    """
    if not texts:
        raise ValueError("`texts` must be a non-empty list of strings.")

    connect_milvus(MILVUS_HOST, MILVUS_PORT)
    try:
        coll = Collection(COLLECTION_NAME)
        coll.load()
    except Exception as e:
        connections.disconnect("default")
        raise RuntimeError(f"Failed to open/load collection `{COLLECTION_NAME}`: {e}")

    try:
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        for idx, txt in enumerate(texts, start=1):
            try:
                txt_emb = get_text_embedding(txt)
                q_vec = txt_emb.tolist()
                results = coll.search([q_vec], "embedding", search_params, limit=top_k, output_fields=["id", "path"])
                print(f"Query #{idx}: {txt!r} -> top {top_k} results:")
                for hit in results[0]:
                    print(f"  distance: {hit.score:.4f}, path: {hit.entity.get('path')}, id: {hit.entity.get('id')}")
            except Exception as e:
                print(f"  Query #{idx} ({txt!r}) failed:", e)
    finally:
        try:
            connections.disconnect("default")
        except Exception:
            pass


def _escape_milvus_string(s: str) -> str:
    """
    將字串中的反斜線與雙引號 escape，方便放入 Milvus expr。
    """
    return s.replace("\\", "\\\\").replace('"', '\\"')


def query_by_path(path_value: str):
    """
    使用 Collection.query 對 `path` 欄位做純欄位查詢（不做向量搜尋）。
    回傳符合條件的實體清單。
    注意：比對字串要與儲存時的格式一致（例如路徑斜線）。
    """
    connect_milvus(MILVUS_HOST, MILVUS_PORT)
    try:
        coll = Collection(COLLECTION_NAME)
        escaped = _escape_milvus_string(path_value)
        expr = f'path == "{escaped}"'
        # 指定回傳欄位，例如 id, path
        print(f"尋找 path: {path_value}, expr: {expr}")
        results = coll.query(expr=expr, output_fields=["id", "path"])
        for hit in results:
            print(f"  path: {hit.get('path')}, id: {hit.get('id')}")
    finally:
        connections.disconnect("default")


def search_with_path_filter(q_vec, top_k: int = 5, path_value: str | None = None):
    """
    在向量搜尋時加入 `expr` 過濾 `path` 欄位（若 path_value 為 None 則不加過濾）。
    q_vec: 單一查詢向量 (list or numpy -> 轉成 list)
    """
    connect_milvus(MILVUS_HOST, MILVUS_PORT)
    try:
        coll = Collection(COLLECTION_NAME)
        coll.load()
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        expr = None
        if path_value is not None:
            expr = f'path == "{path_value}"'
        results = coll.search([q_vec], "embedding", search_params, limit=top_k, expr=expr, output_fields=["path"])
        return results
    finally:
        connections.disconnect("default")

if __name__ == "__main__":
    vertex_main()
