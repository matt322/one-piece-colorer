import requests
import os
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# --- Configuration ---
colorurl = "https://scans-hot.planeptune.us/manga/One-Piece-Digital-Colored-Comics/"
bwurl = "https://hot.planeptune.us/manga/One-Piece/"
colorpath = "images/color/"
bwpath = "images/bw/"
headers = {"User-Agent": "Mozilla/5.0"}
max_threads = 8
num_chapters = 1165

# --- Setup directories ---
os.makedirs(bwpath, exist_ok=True)
os.makedirs(colorpath, exist_ok=True)

# --- Configure Logger ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
    handlers=[
        logging.FileHandler("downloader.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def download_and_convert_to_jpg(image_url, save_path_jpg, session):
    if os.path.exists(save_path_jpg):
        logger.debug(f"Already exists: {save_path_jpg}")
        return 200

    try:
        response = session.get(image_url, headers=headers)
        if response.status_code == 200:
            try:
                img = Image.open(BytesIO(response.content))
                if img.mode == "RGBA":
                    img = img.convert("RGB")
                img.save(save_path_jpg, "JPEG")
                logger.info(f"Downloaded: {save_path_jpg}")
            except KeyboardInterrupt:
                if os.path.exists(save_path_jpg):
                    os.remove(save_path_jpg)
                raise
            except Exception as e:
                logger.error(f"Error converting {image_url}: {e}")
                return 500
        elif response.status_code != 404:
            logger.warning(
                f"Failed to download {image_url} - Status code: {response.status_code}"
            )
        return response.status_code
    except KeyboardInterrupt:
        if os.path.exists(save_path_jpg):
            os.remove(save_path_jpg)
        raise
    except Exception as e:
        logger.error(f"Exception while downloading {image_url}: {e}")
        return 500


def process_chapter(chapter):
    logger.info(f"Starting chapter {chapter:04}")
    session = requests.Session()
    try:
        for base, basepath in [(bwurl, bwpath), (colorurl, colorpath)]:
            image = 1
            while True:
                image_url = f"{base}{chapter:04}-{image:03}.png"
                save_path_jpg = f"{basepath}{chapter:04}-{image:03}.jpg"
                code = download_and_convert_to_jpg(image_url, save_path_jpg, session)
                if code != 200:
                    break
                image += 1
        logger.info(f"Finished chapter {chapter:04}")
    except KeyboardInterrupt:
        logger.warning(f"KeyboardInterrupt during chapter {chapter:04}")
        raise
    finally:
        session.close()


def main():
    logger.info("Download started")
    try:
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = [
                executor.submit(process_chapter, chapter)
                for chapter in range(num_chapters)
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    logger.error(f"Error in thread: {e}")
    except KeyboardInterrupt:
        logger.warning("Download interrupted by user. Exiting cleanly.")
    finally:
        logger.info("Download completed or aborted")


if __name__ == "__main__":
    main()
