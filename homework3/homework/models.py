import torch
import torch.nn.functional as F


class CNNClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        Your code here
        Hint: Base this on yours or HW2 master solution if you'd like.
        Hint: Overall model can be similar to HW2, but you likely need some architecture changes (e.g. ResNets)
        """
        input_channels = 3
        num_classes = 6
        self.layers = torch.nn.Sequential(
            self.__block(input_channels, 32, (7, 1, 3)),
            self.__block(32, 64),
            self.__block(64, 128),
            torch.nn.AvgPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(2048, num_classes)
        )

    def __block(self, in_dim, out_dim, extra=(3, 1, 1), pool=(2, 2)):
      return torch.nn.Sequential(
          torch.nn.Conv2d(in_dim, out_dim, *extra),
          torch.nn.BatchNorm2d(out_dim),
          torch.nn.SiLU(),
          torch.nn.MaxPool2d(*pool),
      )

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        """
        return self.layers(x)


class FCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        Your code here.
        Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """
        base_channels = 32
        input_channels = 3
        output_channels = 1
        self.pool = torch.nn.MaxPool2d(3, 2, 1)
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, base_channels, 3, 1, 1),
            torch.nn.BatchNorm2d(base_channels),
            torch.nn.ReLU(True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(base_channels, base_channels * 2, 3, 1, 1),
            torch.nn.BatchNorm2d(base_channels * 2),
            torch.nn.ReLU(True),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(base_channels * 2, base_channels * 4, 3, 1, 1),
            torch.nn.BatchNorm2d(base_channels * 4),
            torch.nn.ReLU(True),
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(base_channels * 4, base_channels * 8, 3, 1, 1),
            torch.nn.BatchNorm2d(base_channels * 8),
            torch.nn.ReLU(True),
        )
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(base_channels * 8, base_channels * 16, 3, 1, 1),
            torch.nn.BatchNorm2d(base_channels * 16),
            torch.nn.ReLU(True),
        )
        self.upconv5 = torch.nn.Sequential(
            torch.nn.Conv2d(base_channels * 16, base_channels * 8, 3, 1, 1),
            torch.nn.BatchNorm2d(base_channels * 8),
            torch.nn.ReLU(True),
        )
        self.upconv4 = torch.nn.Sequential(
            torch.nn.Conv2d(base_channels * 16, base_channels * 4, 3, 1, 1),
            torch.nn.BatchNorm2d(base_channels * 4),
            torch.nn.ReLU(True),
        )
        self.upconv3 = torch.nn.Sequential(
            torch.nn.Conv2d(base_channels * 8, base_channels * 2, 3, 1, 1),
            torch.nn.BatchNorm2d(base_channels * 2),
            torch.nn.ReLU(True),
        )
        self.upconv2 = torch.nn.Sequential(
            torch.nn.Conv2d(base_channels * 4, base_channels, 3, 1, 1),
            torch.nn.BatchNorm2d(base_channels),
            torch.nn.ReLU(True),
        )
        self.upconv1 = torch.nn.Sequential(
            torch.nn.Conv2d(base_channels * 2, base_channels, 3, 1, 1),
            torch.nn.BatchNorm2d(base_channels),
            torch.nn.ReLU(True),
        )
        self.classifier = torch.nn.Conv2d(base_channels, output_channels, 1, 1, 0)
        # self.net = torch.nn.Sequential(
        #   torch.nn.Conv2d(input_channels, 32, kernel_size=7, stride=2, padding=3),
        #   torch.nn.ReLU(),
        #   torch.nn.Conv2d(32, 64, kernel_size=7, stride=2, padding=3),
        #   torch.nn.ReLU(),
        #   torch.nn.Conv2d(64, 128, kernel_size=7, stride=2, padding=3),
        #   torch.nn.ReLU(),
        #   torch.nn.Conv2d(128, 256, kernel_size=7, stride=1, padding=3),
        #   torch.nn.ReLU(),
        #   torch.nn.Conv2d(256, 512, kernel_size=7, stride=1, padding=3),
        #   torch.nn.ReLU(),
        #   torch.nn.Conv2d(512, 256, kernel_size=1, stride=1),
        #   torch.nn.ReLU(),
        # #   torch.nn.UpsamplingBilinear2d(scale_factor=2),
        #   torch.nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
        #   torch.nn.ReLU(),
        #   torch.nn.UpsamplingBilinear2d(scale_factor=2),
        #   torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
        #   torch.nn.ReLU(),
        #   torch.nn.UpsamplingBilinear2d(scale_factor=2),
        #   torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
        #   torch.nn.ReLU(),
        #   torch.nn.UpsamplingBilinear2d(scale_factor=2),
        #   torch.nn.Conv2d(32, output_channels, kernel_size=3, stride=1, padding=1),
        # )

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,6,H,W))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool(x1))
        x3 = self.conv3(self.pool(x2))
        x4 = self.conv4(self.pool(x3))
        x5 = self.conv5(self.pool(x4))
        x5 = self.upconv5(x5)
        x5 = torch.nn.functional.upsample(x5, scale_factor=2)
        x4 = self.upconv4(torch.cat([x5, x4], 1))
        x4 = torch.nn.functional.upsample(x4, scale_factor=2)
        x3 = self.upconv3(torch.cat([x4, x3], 1))
        x3 = torch.nn.functional.upsample(x3, scale_factor=2)
        x2 = self.upconv2(torch.cat([x3, x2], 1))
        x2 = torch.nn.functional.upsample(x2, scale_factor=2)
        x1 = self.upconv1(torch.cat([x2, x1], 1))
        return self.classifier(x1)
        # return self.net(x)




model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
