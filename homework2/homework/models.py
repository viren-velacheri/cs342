import torch


class CNNClassifier(torch.nn.Module):
    def __init__(self):
        """
        Your code here
        """
        super().__init__()
        input_channels = 3
        num_classes = 6
        self.conv = torch.nn.Conv2d(input_channels, 16, 7, 2, 3)
        self.cls = torch.nn.Linear(16, num_classes)

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        x = self.conv(x)
        # Global Pooling
        x = x.mean(dim=(2,3))
        return self.cls(x)


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
