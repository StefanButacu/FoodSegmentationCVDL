import torch
import torch.nn as nn
import torchvision.models
import torchvision.transforms.functional as TF
from torchvision.models import resnet50, resnet18
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

from models.unet_from_scratch.Resnet50 import Truncate_Resnet


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv.forward(x)


class TrucnateResNET_UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(TrucnateResNET_UNET, self).__init__()
        self.downs = nn.ModuleList()
        # self.down = torchvision.models.resnet18( weights= torchvision.models.ResNet18_Weights,
        #                                          progress = True,
        #                                          )

        self.model1 = Truncate_Resnet(output_layer='layer1', in_channels = in_channels, out_channels = 64)
        self.model2 = Truncate_Resnet(output_layer='layer2', in_channels = 64, out_channels = 128)
        self.model3 = Truncate_Resnet(output_layer='layer3', in_channels = 128, out_channels = 256)
        self.model4 = Truncate_Resnet(output_layer='layer4', in_channels = 256, out_channels = 512)

        self.ups = nn.ModuleList()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):

        skip_connections = []
        skip_connections1 = []
        x1 = self.model1(x)
        skip_connections1.append(x1)
        x2 = self.model2(x)
        skip_connections1.append(x2)
        x3 = self.model3(x)
        skip_connections1.append(x3)
        x4 = self.model4(x)
        skip_connections1.append(x4)

        # print(skip_connections)
        for down in self.downs:
            x = down(x)
            # skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        # skip_connections = skip_connections[::-1]
        skip_connections1 = skip_connections1[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            # skip_connection = skip_connections[idx // 2]

            # if x.shape != skip_connection.shape:
            #     x = TF.resize(x, size=skip_connection.shape[2:])

            ########################
            skip_connection1 = skip_connections1[idx // 2]
            if x.shape != skip_connection1.shape:
                skip_connection1 = TF.resize(skip_connection1, size=x.shape[2:])

            concatenate_skip_x = torch.cat((skip_connection1, x), dim=1)  # B, C, H, W
            x = self.ups[idx + 1](concatenate_skip_x)

        return self.final_conv(x)
#
# class UNET(nn.Module):
#     def __init__(self):
#         super(UNET, self).__init__()
#         self.resnet = resnet18(pretrained=True)
#         self.enc1 = nn.Sequential(*list(self.resnet.children())[:3])
#         self.enc2 = nn.Sequential(*list(self.resnet.children())[3:6])
#         self.enc3 = nn.Sequential(*list(self.resnet.children())[6:9])
#         self.enc4 = nn.Sequential(*list(self.resnet.children())[9:])
#
#         self.pool = nn.MaxPool2d(2, 2)
#         self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
#         self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
#         self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
#         self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
#
#         self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
#         self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
#         self.conv5 = nn.Conv2d(32, 1, kernel_size=1)
#         self.sig = nn.Sigmoid()
#
#     def forward(self, x):
#         x1 = self.enc1(x)
#         x2 = self.enc2(self.pool(x1))
#         x3 = self.enc3(self.pool(x2))
#         x4 = self.enc4(self.pool(x3))
#
#         x = self.up1(x4)
#         x = torch.cat((x, x3), dim=1)
#         x = self.conv1(x)
#
#         x = self.up2(x)
#         x = torch.cat((x, x2), dim=1)
#         x = self.conv2(x)
#
#         x = self.up3(x)
#         x = torch.cat((x, x1), dim=1)
#         x = self.conv3(x)
#
#         x = self.up4(x)
#         x = self.conv4(x)
#
#         x = self.conv5(x)
#         x = self.sig(x)
#
#         return x


import torch
import torch.nn as nn
from torchvision.models import resnet18


class MyUNET(nn.Module):
    def __init__(self):
        super(MyUNET, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.enc1 = nn.Sequential(*list(self.resnet.children())[:3])
        self.enc2 = nn.Sequential(*list(self.resnet.children())[:6])
        self.enc3 = nn.Sequential(*list(self.resnet.children())[:9])
        self.enc4 = nn.Sequential(*list(self.resnet.children())[:-1])

        self.pool = nn.MaxPool2d(2, 2)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.final_conv = nn.Conv2d(32, 3, kernel_size=1)


    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x)
        x3 = self.enc3(x)
        x4 = self.enc4(x)

        x = self.up1(x4)

        x3 = TF.resize(x3, size=x.shape[2:])
        x = torch.cat((x, x3), dim=1)


        x = self.conv1(x)

        x = self.up2(x)

        x2 = TF.resize(x2, size=x.shape[2:])
        x = torch.cat((x, x2), dim=1)
        x = self.conv2(x)

        x = self.up3(x)

        x1 = TF.resize(x1, size=x.shape[2:])
        x = torch.cat((x, x1), dim=1)
        x = self.conv3(x)

        x = self.up4(x)
        x = self.conv4(x)

        x = self.final_conv(x)

        return x


def test():
    x = torch.randn((1, 3, 256, 256))
    # model = TrucnateResNET_UNET()
    # pred = model(x)
    myModel = MyUNET()
    from torchsummary import summary
    summary(myModel,input_size=(3, 512, 512))
    print(x.shape)
    # print(pred.shape)
    # assert x.shape == pred.shape


if __name__ == '__main__':
    test()
