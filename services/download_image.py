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
        logger.warning(f"download_image failed for {url}: {e}")
        return None


def _download_and_time(prod: ProductRecord) -> tuple[int, bytes | None, float]:
    t0 = time.perf_counter()
    img = download_image(prod.pic)
    elapsed = time.perf_counter() - t0
    # logger.info(f"download_image took {elapsed:.2f} seconds for productId: {prod.product_id}")
    return prod.product_id, img, elapsed


from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# module-level reusable session with retries
_HTTP_SESSION = None
def _init_http_session(retries: int = 3, backoff_factor: float = 0.5, status_forcelist=(500,502,503,504)):
    global _HTTP_SESSION
    if _HTTP_SESSION is not None:
        return _HTTP_SESSION
    session = requests.Session()
    retries_cfg = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=list(status_forcelist),
        allowed_methods=frozenset(['GET', 'POST', 'HEAD', 'OPTIONS'])
    )
    adapter = HTTPAdapter(max_retries=retries_cfg)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    # common headers to avoid some CDN blocks
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; product-embed-bot/1.0)",
        "Accept": "image/*,*/*;q=0.8"
    })
    _HTTP_SESSION = session
    return _HTTP_SESSION


def download_image2(url: str, timeout: int = 10, retries: int = 3) -> Optional[bytes]:
    """
    Robust image downloader using a session with retries and exponential backoff.
    Returns bytes on success or None on final failure. Logs attempts.
    """
    session = _init_http_session(retries=retries)
    try:
        logger.debug("download_image: starting GET %s", url)
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if not content_type.startswith("image/"):
            logger.warning("download_image: URL returned non-image Content-Type=%s for %s", content_type, url)
            return None
        return resp.content
    except Exception as e:
        logger.warning("download_image failed for %s: %s", url, e)
        return None


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
