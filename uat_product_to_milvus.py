import logging
import time
from typing import List, Dict

from pymilvus import connections, Collection

from services.milvus_client import connect_milvus, init_milvus_collection, print_collection_stats
from services.download_image import download_image, batch_download_images
from services.google_multimodal_embeddings import embed_text_and_image, batch_get_embeddings
from services.product_query import fetch_products, fetch_product_by_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

MILVUS_HOST = '127.0.0.1'
MILVUS_PORT = 19530
COLLECTION_NAME = 'uat_product_embeddings'


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
            logger.warning(f"skip product {prod.product_id} due to missing image")
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
    print_collection_stats(COLLECTION_NAME)


async def batch_process_and_store_all(batch_size: int = 64, concurrency: int = 24):
    process_and_store_all_t0 = time.perf_counter()
    products = fetch_products()
    logger.info(f"fetched {len(products)} products")

    if not products:
        logger.info("no products to process")
        return

    embed_dim = 1408  # 預設向量維度需與模型相符
    coll = init_milvus_collection(COLLECTION_NAME, embed_dim, MILVUS_HOST, MILVUS_PORT)

    total = len(products)
    total_batches = (total + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        batch = products[start:start + batch_size]
        logger.info(f"processing batch {batch_idx + 1}/{total_batches}: {len(batch)} products (concurrency={concurrency})")

        t0 = time.perf_counter()
        images_by_id = await batch_download_images(batch, max_workers=10)
        t_download = time.perf_counter() - t0
        logger.info(f"batch {batch_idx + 1} download finished in {t_download:.2f} seconds")

        t0 = time.perf_counter()
        embeddings = await batch_get_embeddings(batch, images_by_id, max_workers=concurrency)
        t_embed = time.perf_counter() - t0
        logger.info(f"batch {batch_idx + 1} embedding finished in {t_embed:.2f} seconds")

        # prepare insert lists
        docs: List[Dict] = []
        for prod in batch:
            vec = embeddings.get(prod.product_id)
            if not vec:
                logger.warning(f"skip product {prod.product_id} due to missing embedding")
                continue
            if len(vec) != embed_dim:
                logger.warning(f"vector dim mismatch for {prod.product_id}: got {len(vec)} expected {embed_dim}")
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

    logger.info(f"{__name__} completed in {time.perf_counter() - process_and_store_all_t0:.2f} seconds")
    print_collection_stats(COLLECTION_NAME)


def search_milvus(product_id: int, top_k: int = 5):
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
    batch_process_and_store_all()
