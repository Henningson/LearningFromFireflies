import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, encoder, out_channels, features):
        super(Decoder, self).__init__()
        self.ups = nn.ModuleList()
        self.encoder = encoder
        self.out_channels = out_channels

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

    def forward(self, x):
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = self.encoder.skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, features):
        super(Encoder, self).__init__()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.in_channels = in_channels

        # Downsampling
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

    def forward(self, x):
        self.skip_connections = []
        for down in self.downs:
            x = down(x)
            self.skip_connections.append(x)
            x = self.pool(x)

        self.skip_connections = self.skip_connections[::-1]

        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=3,
        features=[32, 64, 128, 256, 512],
        state_dict=None,
    ):
        super(UNet, self).__init__()

        self.bottleneck_size = features[-1] * 2

        self.encoder = Encoder(in_channels, features)
        self.decoder = Decoder(self.encoder, out_channels, features)
        self.bottleneck = DoubleConv(features[-1], self.bottleneck_size)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        if state_dict:
            self.load_from_dict(state_dict)

    def get_statedict(self):
        return {
            "Encoder": self.encoder.state_dict(),
            "Bottleneck": self.bottleneck.state_dict(),
            "Decoder": self.decoder.state_dict(),
            "LastConv": self.final_conv.state_dict(),
        }

    def load_from_dict(self, dict):
        self.encoder.load_state_dict(dict["Encoder"])
        self.bottleneck.load_state_dict(dict["Bottleneck"])
        self.decoder.load_state_dict(dict["Decoder"])

        try:
            self.final_conv.load_state_dict(dict["LastConv"])
        except:
            print("Final conv not initialized.")

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)

        return self.final_conv(x)


def test():
    batch_size = 16
    x = torch.randn((batch_size, 1, 256, 256)).cuda()
    # y = torch.randn((batch_size, 4, 100)).cuda()
    model = UNet().cuda()
    test = model(x)
    print(test)


if __name__ == "__main__":
    test()
