import torch.nn as nn
import torch.nn.functional as F
import torch


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            # batchnorm2d is when batchnorm calculated across channels,no math difference
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.25)
        )

    def forward(self, x):
        return self.block(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, height, width = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Cat_Dog(nn.Module):

    def __init__(self):
        super(Cat_Dog, self).__init__()

        self.cnn = nn.Sequential(
            ConvBlock(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),

            ConvBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            SELayer(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            ConvBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            SELayer(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            ConvBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0),
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(128, 64, kernel_size=1, stride=1),
        )

        self.clc_layer = nn.Sequential(nn.Linear(25600, 200),
                                       nn.Linear(200, 2))

    def forward(self, x):
        x = self.cnn(x)
        x = F.avg_pool2d(x, kernel_size=2)
        x = self.out_conv(x)
        x = x.view(x.size(0), -1)
        x = self.clc_layer(x)
        return x


if __name__ == "__main__":
    input = torch.rand(16, 3, 160, 160).cuda()
    model = Cat_Dog()
    model.cuda()
    output = model(input)
    #print(output.shape)
