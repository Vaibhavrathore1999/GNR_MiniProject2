import torch
import torch.nn as nn

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, bias=True, norm=False, relu=True):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False
        padding = kernel_size // 2
        layers = [nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias)]
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.conv1 = BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True)
        self.conv2 = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))

class EncoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(EncoderBlock, self).__init__()
        self.conv1 = BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True)
        self.resblock = ResBlock(out_channel, out_channel)
        self.conv2 = BasicConv(out_channel, out_channel, kernel_size=3, stride=2, relu=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.resblock(x)
        return self.conv2(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channel, in_channel // 2, kernel_size=2, stride=2)
        self.conv1 = BasicConv(in_channel // 2, out_channel, kernel_size=3, stride=1, relu=True)
        self.resblock = ResBlock(out_channel, out_channel)
        self.conv2 = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x, skip_connection):
        x = self.up(x)
        if skip_connection is not None:
            x = torch.cat([x, skip_connection], dim=1)
        x = self.conv1(x)
        x = self.resblock(x)
        return self.conv2(x)

class DeblurModel(nn.Module):
    def __init__(self, num_channels=3, num_res=8):
        super(DeblurModel, self).__init__()
        self.encoders = nn.ModuleList([
            EncoderBlock(num_channels, 64),
            EncoderBlock(64, 128),
            EncoderBlock(128, 256)
        ])
        self.decoders = nn.ModuleList([
            DecoderBlock(256, 128),
            DecoderBlock(128, 64),
            DecoderBlock(64, num_channels)
        ])

    def forward(self, x):
        skip_connections = []
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
        for i, decoder in enumerate(self.decoders):
            skip = skip_connections.pop() if i < len(skip_connections) else None
            x = decoder(x, skip)
        return x

# Example usage:
deblur_model = DeblurModel()
# print(deblur_model)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

total_params = count_parameters(deblur_model)
print(f"Total number of parameters: {total_params}")