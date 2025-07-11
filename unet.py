import torch
import torch.nn as nn
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
from skimage import color
import numpy as np


class Block(nn.Module):
    def __init__(self, prev_chans, chans, down=True, use_trans = True, use_attn = False):
        super().__init__()
        self.c2 = nn.Conv2d(in_channels=chans, out_channels=chans, kernel_size=3, padding=1)
        self.silu = nn.SiLU()
        self.norm1 = nn.GroupNorm(32, chans)
        self.norm2 = nn.GroupNorm(32, chans)
        self.use_attn = use_attn
        self.use_trans = use_trans
        self.dropout = nn.Dropout(0.1)
        if use_attn:
            self.attn = Attention(chans)

        if down:
            self.trans = nn.Conv2d(chans, chans, 4, 2, 1)
            self.c1 = nn.Conv2d(in_channels=prev_chans, out_channels=chans, kernel_size=3, padding=1)
            if prev_chans != chans:
                self.shortcut = nn.Conv2d(prev_chans, chans, 1)
            else:
                self.shortcut = nn.Identity()
        else:
            self.trans = nn.ConvTranspose2d(chans, chans, 4, 2, 1)
            self.c1 = nn.Conv2d(in_channels=prev_chans*2, out_channels=chans, kernel_size=3, padding=1)
            if prev_chans*2 != chans:
                self.shortcut = nn.Conv2d(prev_chans*2, chans, 1)
            else:
                self.shortcut = nn.Identity()


    def forward(self, x, residual=None):
        if residual == None:
            h = self.dropout(self.norm1(self.silu(self.c1(x))))
        else:
            x = torch.cat((x, residual), dim=1)
            h = self.dropout(self.norm1(self.silu(self.c1(x))))

        h = self.norm2(self.dropout(self.silu(self.c2(h))))
        h += self.shortcut(x)
        if self.use_attn:
            h = self.attn(h)
        if self.use_trans:
            return self.trans(h), h
        return h, None

class Attention(nn.Module):
    def __init__(self, n_channels: int, n_heads: int = 4, d_k: int = None, n_groups: int = 32):
        super().__init__()
        if d_k is None:
            d_k = n_channels
        self.norm = nn.GroupNorm(n_groups, n_channels)
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        self.output = nn.Linear(n_heads * d_k, n_channels)
        self.scale = d_k ** -0.5
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        batch_size, n_channels, height, width = x.shape
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        attn = attn.softmax(dim=2)
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        res = res.reshape(batch_size, -1, self.n_heads * self.d_k)
        res = self.output(res)
        res += x
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)
        return res

class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        c = 32
        down_channels = (c, c, 2*c, 2*c, 4*c, 4*c)
        up_channels = (4*c, 4*c, 2*c, 2*c, c, c)
        use_attn = (False, False, False, False)
        self.downs = nn.ModuleList([
                                    Block(down_channels[0], down_channels[0], use_attn=use_attn[0], use_trans=False),
                                    Block(down_channels[0], down_channels[1], use_attn=use_attn[0]),
                                    Block(down_channels[1], down_channels[2], use_attn=use_attn[1], use_trans=False),
                                    Block(down_channels[2], down_channels[3], use_attn=use_attn[1]),
                                    Block(down_channels[3], down_channels[4], use_attn=use_attn[2], use_trans=False),
                                    Block(down_channels[4], down_channels[5], use_attn=use_attn[2])
                                    ])
        self.ups = nn.ModuleList([
                                    Block(up_channels[0], up_channels[1], use_attn=use_attn[2], use_trans=False, down=False),
                                    Block(up_channels[1]//2, up_channels[2], use_attn=use_attn[2], down=False),
                                    Block(up_channels[2], up_channels[3], use_attn=use_attn[1], use_trans=False, down=False),
                                    Block(up_channels[3]//2, up_channels[4], use_attn=use_attn[1], down=False),
                                    Block(up_channels[4], up_channels[5], use_attn=use_attn[0], use_trans=False, down=False),
                                    Block(up_channels[5]//2, up_channels[5], use_attn=use_attn[0], down=False, use_trans=False)
                                    ])
        self.output = nn.Conv2d(up_channels[-1], 3, 1)
        self.start = nn.Conv2d(1, c, 3, padding=1)
        self.middle = nn.ModuleList([
            Block(4*c, 8*c, use_attn=False, use_trans=False),
            Block(4*c, 4*c, use_attn=False, down=False)
        ])
        self.conversion = LabToRGB()

    def forward(self, x):
        x = self.start(x)
        residuals = []
        for i,d in enumerate(self.downs):
            x = d(x)
            if i % 2 == 1:
                residuals.append(x[1])
            x = x[0]
        x = self.middle[0](x)[0]
        x = self.middle[1](x)[0]
        for i,d in enumerate(self.ups):
            if i % 2 == 0:
                x = d(x, residuals.pop())[0]
            else:
                x = d(x)[0]
        return self.conversion(self.output(x))

class LabToRGB(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, lab: torch.Tensor) -> torch.Tensor:
        """
        Convert LAB image tensor to RGB.
        Input/output tensors should be in range [-1, 1].
        Shape: [B, 3, H, W]
        """
        # 1. Rescale LAB from [-1, 1] to real LAB ranges
        L = (lab[:, 0:1] + 1) * 50        # [-1, 1] → [0, 100]
        a = lab[:, 1:2] * 127             # [-1, 1] → [-127, 127]
        b = lab[:, 2:3] * 127             # [-1, 1] → [-127, 127]

        # 2. LAB → XYZ
        fy = (L + 16.0) / 116.0
        fx = fy + (a / 500.0)
        fz = fy - (b / 200.0)

        # cube or linear branch
        epsilon = 6 / 29
        fx3 = torch.where(fx > epsilon, fx ** 3, (116 * fx - 16) / 903.3)
        fy3 = torch.where(fy > epsilon, fy ** 3, (116 * fy - 16) / 903.3)
        fz3 = torch.where(fz > epsilon, fz ** 3, (116 * fz - 16) / 903.3)

        # Reference white D65
        X = fx3 * 0.95047
        Y = fy3 * 1.00000
        Z = fz3 * 1.08883

        # 3. XYZ → RGB
        r = X *  3.2406 + Y * -1.5372 + Z * -0.4986
        g = X * -0.9689 + Y *  1.8758 + Z *  0.0415
        b = X *  0.0557 + Y * -0.2040 + Z *  1.0570
        rgb = torch.cat([r, g, b], dim=1)

        # 4. Apply gamma correction
        rgb = torch.where(rgb > 0.0031308,
                          1.055 * torch.pow(rgb.clamp(min=1e-6), 1 / 2.4) - 0.055,
                          12.92 * rgb)

        # 5. Clamp and rescale to [-1, 1]
        rgb = rgb.clamp(0, 1)
        rgb = rgb * 2 - 1

        return rgb


if __name__ == "__main__":
    img = Image.open("images/color/0001-001.jpg").convert("RGB").resize((256, 256))
    img_np = np.asarray(img) / 255.0  # shape: (H, W, 3), range [0,1]

    # Convert to LAB using skimage
    lab = color.rgb2lab(img_np)  # shape: (H, W, 3)

    # Normalize to [-1, 1]
    lab[:, :, 0] = lab[:, :, 0] / 50.0 - 1.0      # L: [0,100] → [-1,1]
    lab[:, :, 1] = lab[:, :, 1] / 127.0           # a: [-127,127] → [-1,1]
    lab[:, :, 2] = lab[:, :, 2] / 127.0           # b: [-127,127] → [-1,1]

    # Convert to torch tensor
    lab_tensor = torch.from_numpy(lab).permute(2, 0, 1).unsqueeze(0).float()  # [1, 3, H, W]

    # === Convert back to RGB ===
    converter = LabToRGB()
    rgb_tensor = converter(lab_tensor).clamp(-1, 1)

    # Denormalize back to [0, 1] for display
    print(rgb_tensor.shape, lab_tensor.shape)
    rgb_out = (rgb_tensor.squeeze(0).permute(1, 2, 0).detach().numpy() + 1) / 2.0
    rgb_out = np.clip(rgb_out, 0, 1)

    # === Display original and reconstructed images ===
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(img_np)
    axs[0].set_title("Original RGB")
    axs[0].axis("off")

    axs[1].imshow(rgb_out)
    axs[1].set_title("Reconstructed RGB (LAB → RGB)")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()