import torch
import torchvision
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
        layers=[32,64,128]
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
        self.relu = torch.nn.ReLU()
        self.norm = torch.nn.BatchNorm2d(3)
        self.down1 = torch.nn.Conv2d(3,32, kernel_size=3, stride=2, padding=1)
        self.down2 = torch.nn.Sequential(
          torch.nn.Dropout2d(p=0.3),
          torch.nn.BatchNorm2d(32),
          torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        )
        self.down3 = torch.nn.Sequential(
          torch.nn.Dropout2d(p=0.3),
          torch.nn.BatchNorm2d(64),
          torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        )
        self.up3 = torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.up2 = torch.nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.up1 = torch.nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.linear = torch.nn.Conv2d(16, 5, kernel_size=1, stride=1, padding=0)
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
        
        h = x.shape[2]
        w = x.shape[3]
        if x.shape[2] < 16 or x.shape[3] < 16:
          hpad = 0
          vpad = 0
          if x.shape[3] < 16:
            hpad = (16 - x.shape[3]) // 2
          if x.shape[2] < 16:
            vpad = (16 - x.shape[2]) // 2
          x = torch.nn.functional.pad(x, (hpad, vpad, vpad))
        
        x = self.norm(x)
        down1 = self.relu(self.down1(x))
        down2 = self.relu(self.down2(down1))
        down3 = self.relu(self.down3(down2))
        up3 = self.relu(self.up3(down3))
        up2 = self.relu(self.up2(up3))
        up1 = self.relu(self.up1(up2))
        up1 = up1[:, :, :h, :w]

        return self.linear(up1)




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
