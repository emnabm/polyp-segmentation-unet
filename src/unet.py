import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()
        c = base
        self.pool = nn.MaxPool2d(2)
        # Encoder
        self.enc1 = DoubleConv(in_ch, c)
        self.enc2 = DoubleConv(c, c*2)
        self.enc3 = DoubleConv(c*2, c*4)
        self.enc4 = DoubleConv(c*4, c*8)
        # Bottleneck
        self.bott = DoubleConv(c*8, c*16)
        # Decoder
        self.up1 = nn.ConvTranspose2d(c*16, c*8, 2, stride=2); self.dec1 = DoubleConv(c*16, c*8)
        self.up2 = nn.ConvTranspose2d(c*8,  c*4, 2, stride=2); self.dec2 = DoubleConv(c*8,  c*4)
        self.up3 = nn.ConvTranspose2d(c*4,  c*2, 2, stride=2); self.dec3 = DoubleConv(c*4,  c*2)
        self.up4 = nn.ConvTranspose2d(c*2,  c,   2, stride=2); self.dec4 = DoubleConv(c*2,  c)
        # Output
        self.head = nn.Conv2d(c, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bott(self.pool(e4))
        d1 = self.dec1(torch.cat([self.up1(b),  e4], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d2), e2], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d3), e1], dim=1))
        return self.head(d4)