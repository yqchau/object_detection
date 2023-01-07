import torch
import torch.nn as nn


class FCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.networks = nn.Sequential(
            nn.Linear(12288, 128),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(64, 20),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = x.view(x.shape[0], -1)
        y = self.networks(y)
        return y


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_chans,
        out_chans,
        identity_downsample=None,
        stride=1,
        skip_connections=False,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_chans, out_chans, kernel_size=3, padding=1, stride=stride
        )
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_chans)
        self.bn2 = nn.BatchNorm2d(out_chans)
        self.relu = nn.ReLU(inplace=True)
        self.identity_downsample = identity_downsample
        self.skip_connections = skip_connections

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        if self.skip_connections:
            x += identity

        x = self.relu(x)

        return x


class CNN18(nn.Module):
    def __init__(self, in_chans, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=1)
        # self.layer3 = self.__make_layer(128, 256, stride=1)
        # self.layer4 = self.__make_layer(256, 512, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def __make_layer(self, in_chans, out_chans, stride):
        return nn.Sequential(
            BasicBlock(in_chans=in_chans, out_chans=out_chans, stride=stride),
            BasicBlock(in_chans=out_chans, out_chans=out_chans),
        )

    def forward(self, x):

        x = self.conv1(x)  # h/2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # h/4

        x = self.layer1(x)  # h/4
        x = self.layer2(x)  # h/8
        # x = self.layer3(x)  # h/16
        # x = self.layer4(x)  # h/32

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return torch.sigmoid(x)


class CNN18_Backbone(nn.Module):
    def __init__(self, in_chans):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=1)
        self.layer3 = self.__make_layer(128, 1, stride=1)
        # self.layer4 = self.__make_layer(256, 512, stride=1)

    def __make_layer(self, in_chans, out_chans, stride):
        return nn.Sequential(
            BasicBlock(in_chans=in_chans, out_chans=out_chans, stride=stride),
            BasicBlock(in_chans=out_chans, out_chans=out_chans),
        )

    def forward(self, x):

        x = self.conv1(x)  # h/2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # h/4

        x = self.layer1(x)  # h/4
        x = self.layer2(x)  # h/8
        x = self.layer3(x)  # h/16
        # x = self.layer4(x)  # h/32

        return x


if __name__ == "__main__":
    model = CNN18_Backbone(3)
    inputs = torch.randn(1, 3, 224, 224)

    outputs = model(inputs)
    print(outputs.shape)
