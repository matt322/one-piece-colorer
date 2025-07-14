import torch
import torch.nn as nn

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

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


class Unet(nn.Module):

    def __init__(self):
        super().__init__()

        self.dconv_down1 = double_conv(1, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, 3, 1)

        self.conversion = LabToRGB()


    def forward(self, x, return_lab=False):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out if return_lab else self.conversion(out)
    
class Block(nn.Module):
    def __init__(self, prev_chans, chans, down=True, use_trans = True, use_attn = False):
        super().__init__()
        self.c2 = nn.Conv2d(in_channels=chans, out_channels=chans, kernel_size=3, padding=1)
        self.silu = nn.SiLU()
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
            self.trans = nn.Upsample(scale_factor=2, mode="bilinear")
            self.c1 = nn.Conv2d(in_channels=prev_chans*2, out_channels=chans, kernel_size=3, padding=1)
            if prev_chans*2 != chans:
                self.shortcut = nn.Conv2d(prev_chans*2, chans, 1)
            else:
                self.shortcut = nn.Identity()


    def forward(self, x, residual=None):
        if residual == None:
            h = self.dropout(self.silu(self.c1(x)))
        else:
            x = torch.cat((x, residual), dim=1)
            h = self.dropout(self.silu(self.c1(x)))

        h = self.dropout(self.silu(self.c2(h)))
        h += self.shortcut(x)
        if self.use_trans:
            return self.trans(h), h
        return h, None


class mattUnet(nn.Module):
    def __init__(self):
        super().__init__()
        c = 32
        down_channels = (c, c, 2*c, 2*c, 4*c, 4*c)
        up_channels = (4*c, 4*c, 2*c, 2*c, c, c)
        use_attn = (False, False, False, False)
        self.downs = nn.ModuleList([
                                    Block(down_channels[0], down_channels[0], use_trans=False),
                                    Block(down_channels[0], down_channels[1]),
                                    Block(down_channels[1], down_channels[2], use_trans=False),
                                    Block(down_channels[2], down_channels[3]),
                                    Block(down_channels[3], down_channels[4], use_trans=False),
                                    Block(down_channels[4], down_channels[5])
                                    ])
        self.ups = nn.ModuleList([
                                    Block(up_channels[0], up_channels[1], use_trans=False, down=False),
                                    Block(up_channels[1]//2, up_channels[2], down=False),
                                    Block(up_channels[2], up_channels[3], use_trans=False, down=False),
                                    Block(up_channels[3]//2, up_channels[4], down=False),
                                    Block(up_channels[4], up_channels[5], use_trans=False, down=False),
                                    Block(up_channels[5]//2, up_channels[5], down=False, use_trans=False)
                                    ])
        self.output = nn.Conv2d(up_channels[-1], 3, 1)
        self.start = nn.Conv2d(1, c, 3, padding=1)
        self.middle = nn.ModuleList([
            Block(4*c, 8*c, use_attn=False, use_trans=False),
            Block(4*c, 4*c, use_attn=False, down=False)
        ])
        self.conversion = LabToRGB()

    def forward(self, x, return_lab=False):
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
        if return_lab:
            return self.output(x)
        else:
            return self.conversion(self.output(x))