import logging
import time
from typing import List, Dict, Any

from pymilvus import connections, Collection

from models.product_record import ProductRecord
import services.my_milvus_client as my_milvus_client
import services.download_image as dl_img
import services.google_multimodal_embeddings as vertex_mm_emb
import services.product_query as product_query

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

MILVUS_HOST = '127.0.0.1'
MILVUS_PORT = 19530
COLLECTION_NAME = 'uat_product_embeddings'
EMBEDDING_DIM = 1408


def process_and_store_all(batch_size: int = 64):
    """
    同步版本，一次處理 batch_size 筆資料。
    下載圖片與取得向量皆為同步呼叫。
    依序處理每個 batch，避免一次載入過多資料。
    適合資料量較小或環境不支援 asyncio 的情況。
    也可作為非同步版本的參考範例。
    """
    process_and_store_all_t0 = time.perf_counter()
    products = product_query.fetch_products()
    ori_len = len(products)
    products = _dedup(products)
    logger.info(f"fetched {ori_len} products, reduced to {len(products)} after deduplication")

    if not products:
        logger.info("no products to process")
        return

    coll = my_milvus_client.init_milvus_collection(COLLECTION_NAME, EMBEDDING_DIM, MILVUS_HOST, MILVUS_PORT)

    total = len(products)
    total_batches = (total + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        batch = products[start:start + batch_size]
        logger.info(f"processing batch {batch_idx + 1}/{total_batches}: {len(batch)} products")

        # prepare insert lists
        docs: List[Dict] = []
        for prod in batch:
            img = None
            if prod.pic:
                pc_start = time.perf_counter()
                img = dl_img.download_image(prod.pic)
                pc_end = time.perf_counter()
                logger.info(f"{dl_img.download_image.__name__} took {pc_end - pc_start:.2f} seconds for productId: {prod.product_id}")
            if not img:
                logger.warning(f"skip product {prod.product_id} due to missing image")
                continue

            try:
                pc_start = time.perf_counter()
                vec = vertex_mm_emb.embed_text_and_image(prod.alias_name, img)
                pc_end = time.perf_counter()
                logger.info(f"{vertex_mm_emb.embed_text_and_image.__name__} took {pc_end - pc_start:.2f} seconds for productId: {prod.product_id}")
            except Exception as e:
                logger.warning(f"embedding failed for {prod.product_id}: {e}")
                continue

            # 檢查向量維度
            if len(vec) != EMBEDDING_DIM:
                logger.warning(f"vector dim mismatch for {prod.product_id}: got {len(vec)} expected {EMBEDDING_DIM}")
                continue

            doc = {
                "product_id": prod.product_id,
                "sale_code": prod.sale_code,
                "alias_name": prod.alias_name,
                "embedding": vec
            }
            docs.append(doc)

        if docs:
            pc_start = time.perf_counter()
            coll.upsert(docs)
            coll.flush()
            pc_end = time.perf_counter()
            logger.info(f'inserted doc count: {len(docs)}, took {pc_end - pc_start:.2f} seconds')
        else:
            logger.info(f"no valid embeddings to insert")

    logger.info(f"{__name__} completed in {time.perf_counter() - process_and_store_all_t0:.2f} seconds")
    my_milvus_client.print_collection_stats(COLLECTION_NAME)


async def process_and_store_all_async(batch_size: int = 64, concurrency: int = 16):
    """"
    非同步版本，一次處理 batch_size 筆資料。
    下載圖片與取得向量皆為非同步呼叫，並可設定最大同時執行數量 concurrency。
    依序處理每個 batch，避免一次載入過多資料。
    適合資料量較大或環境支援 asyncio 的情況。
    也可作為同步版本的參考範例。
    """
    batch_process_and_store_all_t0 = time.perf_counter()
    products = await product_query.fetch_products_async()
    ori_len = len(products)
    products = _dedup(products)
    logger.info(f"fetched {ori_len} products, reduced to {len(products)} after deduplication")

    if not products:
        logger.info("no products to process")
        return

    coll = my_milvus_client.init_milvus_collection(COLLECTION_NAME, EMBEDDING_DIM, MILVUS_HOST, MILVUS_PORT)

    total = len(products)
    total_batches = (total + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        batch = products[start:start + batch_size]
        logger.info(f"processing batch {batch_idx + 1}/{total_batches}: {len(batch)} products (concurrency={concurrency})")

        t0 = time.perf_counter()
        images_by_id = await dl_img.batch_download_images_async(batch, max_workers=concurrency)
        t_download = time.perf_counter() - t0
        logger.info(f"batch {batch_idx + 1} {dl_img.batch_download_images_async.__name__} took {t_download:.2f} seconds")

        t0 = time.perf_counter()
        embeddings = await vertex_mm_emb.batch_get_embeddings(batch, images_by_id, max_workers=concurrency)
        t_embed = time.perf_counter() - t0
        logger.info(f"batch {batch_idx + 1} {vertex_mm_emb.batch_get_embeddings.__name__} took {t_embed:.2f} seconds")

        # prepare insert lists
        docs: List[Dict] = []
        for prod in batch:
            vec = embeddings.get(prod.product_id)
            if not vec:
                logger.warning(f"skip product {prod.product_id} due to missing embedding")
                continue
            if len(vec) != EMBEDDING_DIM:
                logger.warning(f"vector dim mismatch for {prod.product_id}: got {len(vec)} expected {EMBEDDING_DIM}")
                continue

            doc = {
                "product_id": prod.product_id,
                "sale_code": prod.sale_code,
                "alias_name": prod.alias_name,
                "embedding": vec
            }
            docs.append(doc)

        if docs:
            pc_start = time.perf_counter()
            coll.upsert(docs)
            coll.flush()
            pc_end = time.perf_counter()
            logger.info(f'inserted batch {batch_idx + 1} of {len(docs)}, took {pc_end - pc_start:.2f} seconds')
        else:
            logger.info(f"no valid embeddings to insert for batch {batch_idx + 1}")

    logger.info(f"{__name__} completed in {time.perf_counter() - batch_process_and_store_all_t0:.2f} seconds")
    my_milvus_client.print_collection_stats(COLLECTION_NAME)


async def search_milvus_async(product_id: int, top_k: int = 5):
    prod = await product_query.fetch_product_by_id_async(product_id)
    if not prod:
        raise RuntimeError(f"Product with id {product_id} not found")

    logger.info(f'searching Milvus for product_id={prod.product_id}, text="{prod.alias_name}", image_path={prod.pic}')

    my_milvus_client.connect_milvus(MILVUS_HOST, MILVUS_PORT)
    try:
        coll = Collection(COLLECTION_NAME)
        coll.load()
    except Exception as e:
        connections.disconnect("default")
        raise RuntimeError(f"Failed to open/load collection `{COLLECTION_NAME}`: {e}")

    try:
        img = await dl_img.download_image_async(url=prod.pic, session=None)
        if not img:
            logger.warning(f"download_image returned None for {prod.pic}")
            return []

        q_vec = vertex_mm_emb.embed_text_and_image(prod.alias_name, img)
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


def _dedup(products: list[ProductRecord]) -> list[Any]:
    """
    去重，保留第一次出現的 product_id（原序不變）
    """
    seen = set()
    unique_products = []
    for p in products:
        if p.product_id in seen:
            continue
        seen.add(p.product_id)
        unique_products.append(p)
    products = unique_products
    return products


if __name__ == "__main__":
    process_and_store_all_async()
