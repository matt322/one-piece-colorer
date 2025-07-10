import requests
import os
from PIL import Image

colorurl = "https://scans-hot.planeptune.us/manga/One-Piece-Digital-Colored-Comics/"
bwurl = "https://hot.planeptune.us/manga/One-Piece/"
colorpath = "images/color/"
bwpath = "images/bw/"
headers = {
    "User-Agent": "Mozilla/5.0"
}

def convert_png_to_jpg(path):
    img = Image.open(path)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img.save(f"{path[:-4]}.jpg", "JPEG")
    os.remove(path)
        
def download_image_from_url(image_url, save_path, session):
    response = session.get(image_url, headers=headers)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
    elif response.status_code != 404:
        print(f"Failed to download image at {image_url}. Status code: {response.status_code}")
    return response.status_code

def chapterimage(path):
    return int(path[:4]), int(path[5:8])

images = os.listdir(bwpath)
images.sort(key=chapterimage)
most_recent_chapter, most_recent_image = chapterimage(images[-1])

for chapter in range(most_recent_chapter, 1065):
    session = requests.Session()
    image = 1
    if chapter == most_recent_chapter:
        image = most_recent_image + 1
    url = f"{bwurl}{chapter:04}-{image:03}.png"
    while (code := download_image_from_url(url, f"{bwpath}{chapter:04}-{image:03}.png", session)) == 200:
        convert_png_to_jpg(f"{bwpath}{chapter:04}-{image:03}.png")
        image += 1
        url = f"{bwurl}{chapter:04}-{image:03}.png"
    if code not in [200, 404]:
            break
    session.close()
    print(f"Downloaded chapter {chapter:04}")