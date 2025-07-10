from unet import Unet
import torch
import lpips
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt


loss_fn = lpips.LPIPS(net='alex')

model = Unet()

model(torch.randn(1, 3, 1200, 1846))


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


for epoch in range(10):
    model.load_state_dict(torch.load("model.pt"))
    model.train()
    model.cuda()

    for i in range(100):
        color = rbg2lab_tensor(np.array(Image.open(f"images/color/{epoch:04}-{i:03}.jpg")))
        bw = rbg2lab_tensor(np.array(Image.open(f"images/bw/{epoch:04}-{i:03}.jpg")))
        
        with torch.no_grad():
            noise = torch.randn_like(color).cuda()
            noised = torch.clamp(color + noise, 0, 1)
        pred = model(noised, 1000)
        loss = loss_fn(pred, color)
        print(f"{epoch} {i} {loss.item():0.6f}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim = torch.optim.Adam(model.parameters(), lr=1e-4)
        optim.step()
        optim.zero_grad()
    torch.save(model.state_dict(), "model.pt")
    model.eval()
    torch.save(model.state_dict(), f"model-{epoch}.pt")


