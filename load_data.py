import os
from torch.utils.data import Dataset, DataLoader
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

def rbg2lab_tensor(image):
    img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    img_lab = img_lab.astype(np.float32)
    img_lab /= 255.0
    return torch.from_numpy(img_lab).permute(2, 0, 1)

def tensor2image(tensor):
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = (img * 255.0).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    return img

shape = (1066, 1600)

class Opdataset(Dataset):
    def __init__(self, chapters):
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

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        resx = rbg2lab_tensor(np.array(cv2.resize(cv2.imread(self.x[idx]), shape, interpolation=cv2.INTER_CUBIC)))
        resy = rbg2lab_tensor(np.array(cv2.resize(cv2.imread(self.y[idx]), shape, interpolation=cv2.INTER_CUBIC)))
        return resx, resy

if __name__ == "__main__":
    trainchapters, testchapters = [], []
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
            testchapters.append(ch)
        else:
            trainchapters.append(ch)

    print(len(trainchapters), len(testchapters))

    trainset = Opdataset(trainchapters)
    x, y = trainset[0]
    plt.imshow(tensor2image(y))
    plt.show()



