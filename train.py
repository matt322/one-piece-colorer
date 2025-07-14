from unet import Unet, LabToRGB
from dataset import Opdataset, tensor2image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import lpips
import matplotlib.pyplot as plt
import os

# --- Set up paths ---
IMAGE_PATH = f"training_images/run-{len(os.listdir('training_images')) + 1}/"
os.mkdir(IMAGE_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Discriminator ---
class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        def block(in_c, out_c, stride):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, stride, 1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True)
            )
        self.model = nn.Sequential(
            block(input_channels, 64, 2),
            block(64, 128, 2),
            block(128, 256, 2),
            nn.Conv2d(256, 1, 4, 1, 0)  # Final patch output
        )

    def forward(self, x):
        return self.model(x)

# --- Filter chapters ---
def filter_chapters():
    trainchapters, testchapters = [], []
    chapters = sorted(os.listdir("images/colorch"))[1:]
    for ch in chapters:
        bw = len(os.listdir(f"images/bwch/{ch}"))
        color = len(os.listdir(f"images/colorch/{ch}"))
        if ".DS_Store" in os.listdir(f"images/bwch/{ch}"):
            os.remove(f"images/bwch/{ch}/.DS_Store")
        if ".DS_Store" in os.listdir(f"images/colorch/{ch}"):
            os.remove(f"images/colorch/{ch}/.DS_Store")
        if bw != color:
            testchapters.append(ch)
        else:
            trainchapters.append(ch)
    print(len(trainchapters), len(testchapters))
    return trainchapters, testchapters

# --- Load data ---
trainchapters, testchapters = filter_chapters()
trainset = Opdataset(trainchapters)
trainset.setmode("RGB")
train = DataLoader(trainset, batch_size=20, shuffle=True)
print(f"Training on {len(trainset)} images")

# --- Losses ---
lpips_loss = lpips.LPIPS(net='alex').to(device)
similarity_loss = lambda x, y: torch.mean(lpips_loss(x, y))
bce = nn.BCEWithLogitsLoss()

# --- Models ---
generator = Unet().to(device)
generator.load_state_dict(torch.load('models/model-7-9.pt', weights_only=True))
discriminator = Discriminator().to(device)

# --- Optimizers ---
opt_G = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

# --- Training loop ---
for epoch in range(30):
    generator.train()
    discriminator.train()
    accumulate_loss = 0
    for idx, batch in enumerate(train):
        bw, real_color = batch
        bw = bw.to(device)
        real_color = real_color.to(device)

        valid = torch.ones((bw.size(0), 1, 1, 1), device=device)
        fake = torch.zeros((bw.size(0), 1, 1, 1), device=device)

        # --- Train Discriminator ---
        fake_color = generator(bw, return_lab=False)
        real_pred = discriminator(real_color)
        fake_pred = discriminator(fake_color.detach())
        loss_D = 0.5 * (bce(real_pred, valid.expand_as(real_pred)) + bce(fake_pred, fake.expand_as(fake_pred)))
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # --- Train Generator ---
        pred_fake = discriminator(fake_color)
        adv_loss = bce(pred_fake, valid.expand_as(pred_fake))
        sim_loss = similarity_loss(fake_color, real_color)
        loss_G = sim_loss + 0.5 * adv_loss  # weight GAN loss lower
        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        accumulate_loss += loss_G.item()
        if idx == 0:
            print(f"memory allocated: {torch.cuda.memory_allocated(0)/(1024**3):.2f} GB")
        if idx % 25 == 0:
            print(f"{epoch} {idx} Loss_G={accumulate_loss/25:.6f} Loss_D={loss_D.item():.6f}")
            accumulate_loss = 0
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(tensor2image(bw[0]), cmap="gray")
            axes[1].imshow(tensor2image(fake_color[0]))
            axes[2].imshow(tensor2image(real_color[0]))
            plt.savefig(f"{IMAGE_PATH}epoch-{epoch}-{idx}.png")
            plt.close(fig)

    generator.eval()
    torch.save(generator.state_dict(), f"models/model-gan-{epoch}.pt")
