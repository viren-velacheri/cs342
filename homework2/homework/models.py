import torch


class CNNClassifier(torch.nn.Module):
    def __init__(self):
        """
        Your code here
        """
        super().__init__()
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
        # self.conv = torch.nn.Conv2d(input_channels, 16, 7, 2, 3)
        # self.cls = torch.nn.Linear(16, num_classes)
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
        """
        # x = self.conv(x)
        # # Global Pooling
        # x = x.mean(dim=(2,3))
        # return self.cls(x)
        return self.layers(x)


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, CNNClassifier):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'cnn.th'))
    raise ValueError("model type '%s' not supported!"%str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = CNNClassifier()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th'), map_location='cpu'))
    return r
