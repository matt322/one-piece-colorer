import os
import matplotlib.pyplot as plt
from PIL import Image


chapters = sorted(os.listdir("images/colorch"))[1:]
i = 0
for ch in chapters:
    bw = len(os.listdir(f"images/bwch/{ch}"))
    color = len(os.listdir(f"images/colorch/{ch}"))
    if ".DS_Store" in os.listdir(f"images/bwch/{ch}"):
        os.remove(f"images/bwch/{ch}/.DS_Store")
        bw -= 1
    if ".DS_Store" in os.listdir(f"images/colorch/{ch}"):
        os.remove(f"images/colorch/{ch}/.DS_Store")
        color -= 1
    if bw != color:
        print(f"Chapter {ch} has {bw} bw images and {color} color images.")
        i += 1
        final = min(bw, color)
        im1, im2 = Image.open(f"images/colorch/{ch}/{sorted(os.listdir(f"images/colorch/{ch}"))[final-1]}"), Image.open(f"images/bwch/{ch}/{sorted(os.listdir(f"images/bwch/{ch}"))[final-1]}")
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(im1)
        axes[1].imshow(im2)
        plt.title(ch)
        plt.show()
print(i)