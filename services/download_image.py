import aiohttp
import asyncio
import logging
import time
from typing import Optional, List, Dict

import requests

from models.product_record import ProductRecord

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def download_image(url: str, timeout: int = 10) -> Optional[bytes]:
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        logger.warning(f"{download_image.__name__} failed for {url}: {e}")
        return None


def _download_and_time(prod: ProductRecord) -> tuple[int, bytes | None, float]:
    t0 = time.perf_counter()
    img = download_image(prod.pic)
    elapsed = time.perf_counter() - t0
    return prod.product_id, img, elapsed


async def batch_download_images(products: List[ProductRecord], max_workers: int = 8) -> dict[int, bytes | None]:
    """
    使用 Semaphore + asyncio.to_thread 限制同時執行數量，避免顯式建立 ThreadPoolExecutor。
    回傳 product_id -> PIL.Image (或 None) 的映射。
    """
    sem = asyncio.Semaphore(max_workers)

    async def worker(prod):
        async with sem:
            # _download_and_time 是同步函式，放到 thread 中執行
            return await asyncio.to_thread(_download_and_time, prod)

    tasks = [asyncio.create_task(worker(p)) for p in products]
    results: Dict[int, Optional[bytes]] = {}
    for fut in asyncio.as_completed(tasks):
        product_id, img, elapsed = await fut
        results[product_id] = img
    return results


async def download_image_async(session: Optional[aiohttp.ClientSession], url: str, timeout: int = 10) -> Optional[bytes]:
    own_session = False
    if session is None:
        session = aiohttp.ClientSession()
        own_session = True

    try:
        async with session.get(url, timeout=timeout) as resp:
            resp.raise_for_status()
            return await resp.read()
    except aiohttp.ClientError as e:
        logger.warning(f"{download_image_async.__name__} failed for {url}: {e}")
        return None
    finally:
        if own_session:
            await session.close()


async def download_and_time_async(prod: ProductRecord, session: aiohttp.ClientSession) -> tuple[int, Optional[bytes], float]:
    t0 = time.perf_counter()
    img = await download_image_async(session, prod.pic)
    elapsed = time.perf_counter() - t0
    return prod.product_id, img, elapsed


async def batch_download_images_async(products: List[ProductRecord], max_workers: int = 8) -> Dict[int, Optional[bytes]]:
    """
    非同步版：重用 aiohttp.ClientSession，使用 Semaphore 控制同時下載數量。
    回傳 product_id -> bytes 或 None 的字典。
    """
    sem = asyncio.Semaphore(max_workers)
    result: Dict[int, Optional[bytes]] = {}

    async with aiohttp.ClientSession() as session:
        async def worker(prod: ProductRecord):
            async with sem:
                return await download_and_time_async(prod, session)

        tasks = [asyncio.create_task(worker(p)) for p in products]
        for fut in asyncio.as_completed(tasks):
            product_id, img, elapsed = await fut
            result[product_id] = img

    return result