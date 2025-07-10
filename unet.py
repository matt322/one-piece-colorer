import torch
import torch.nn as nn


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
        use_attn = (False, False, True, True)
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
        self.start = nn.Conv2d(3, c, 3, padding=1)
        self.middle = nn.ModuleList([
            Block(4*c, 8*c, use_attn=True, use_trans=False),
            Block(4*c, 4*c, use_attn=False, down=False)
        ])

    def forward(self, x, t):
        x = self.start(x)
        residuals = []
        for i,d in enumerate(self.downs):
            x = d(x, t)
            if i % 2 == 1:
                residuals.append(x[1])
            x = x[0]

        x = self.middle[0](x, t)[0]
        x = self.middle[1](x, t)[0]
        for i,d in enumerate(self.ups):
            if i % 2 == 0:
                x = d(x, t, residuals.pop())[0]
            else:
                x = d(x, t)[0]
        return self.output(x)

