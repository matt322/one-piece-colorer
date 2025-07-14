import os
import cv2
from torch.utils.data import Dataset
import torch
import numpy as np

class Opdataset(Dataset):
    def __init__(self, chapters):
        self.shape = (536, 800)
        self.mode = "RGB"
        self.x, self.y = [], []
        for ch in chapters:
            bw = len(os.listdir(f"images/bwch/{ch}"))
            color = len(os.listdir(f"images/colorch/{ch}"))
            if ".DS_Store" in os.listdir(f"images/bwch/{ch}"):
                os.remove(f"images/bwch/{ch}/.DS_Store")
                bw -= 1
            if ".DS_Store" in os.listdir(f"images/colorch/{ch}"):
                os.remove(f"images/colorch/{ch}/.DS_Store")
                color -= 1
            final = min(bw, color)
            for img in range(final):
                xpath = sorted(os.listdir(f"images/bwch/{ch}"))[img]
                ypath = sorted(os.listdir(f"images/colorch/{ch}"))[img]
                self.x.append(f"images/bwch/{ch}/{xpath}")
                self.y.append(f"images/colorch/{ch}/{ypath}")

    def setmode(self, mode):
        if mode not in ["RGB", "LAB"]:
            raise ValueError("Invalid mode")
        self.mode = mode


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        resx = cv2.imread(self.x[idx], cv2.IMREAD_GRAYSCALE)
        if self.mode == "RGB":
            resy = cv2.cvtColor(cv2.imread(self.y[idx]), cv2.COLOR_BGR2RGB)
        else:
            resy = cv2.cvtColor(cv2.imread(self.y[idx]), cv2.COLOR_BGR2LAB)
        resx = np.array(cv2.resize(resx, self.shape, interpolation=cv2.INTER_CUBIC), dtype=np.float32)
        resy = np.array(cv2.resize(resy, self.shape, interpolation=cv2.INTER_CUBIC), dtype=np.float32)
        resx = np.expand_dims(resx, axis=2)
        resx = resx / 127.5 - 1
        resy = resy / 127.5 - 1 #cvtcolor uses 0-255 for uint8 and lab normal ranges for float
        resx = torch.from_numpy(resx).permute(2, 0, 1)
        resy = torch.from_numpy(resy).permute(2, 0, 1)
        return resx, resy

def tensor2image(tensor):
    img = tensor.detach().permute(1, 2, 0).cpu().numpy()
    img = ((img + 1) * 127.5).astype(np.uint8)
    return img