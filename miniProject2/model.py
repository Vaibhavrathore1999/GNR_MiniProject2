import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)

class ResBlock(nn.Module):
    """Residual Block"""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels // 2, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(channels // 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels // 2, channels // 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels // 2)
        self.conv3 = nn.Conv2d(channels // 2, channels, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu(out)
        return out

class DeblurModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.res1 = nn.Sequential(*[ResBlock(64) for _ in range(2)])
        self.down2 = Down(64, 128)
        self.res2 = nn.Sequential(*[ResBlock(128) for _ in range(2)])
        self.down3 = Down(128, 256)
        self.res3 = nn.Sequential(*[ResBlock(256) for _ in range(2)])
        self.up1 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.up3 = Up(64, 32)
        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.res1(x2)
        x3 = self.down2(x2)
        x3 = self.res2(x3)
        x4 = self.down3(x3)
        x4 = self.res3(x4)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        output = self.outc(x)
        return output

# Calculate the total number of parameters
model = DeblurModel()
# torchinfo.summary(model,(16,3,256,448))