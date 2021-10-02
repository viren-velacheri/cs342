import torch
import torch.nn.functional as F


class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        """
        Compute mean(-log(softmax(input)_label))

        @input:  torch.Tensor((B,C))
        @target: torch.Tensor((B,), dtype=torch.int64)

        @return:  torch.Tensor((,))

        Hint: Don't be too fancy, this is a one-liner
        """
        return F.cross_entropy(input, target)


class LinearClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = 3 * 64 * 64
        output_dim = 6
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        return self.linear(x.view(x.shape[0], -1))


class MLPClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = 3 * 64 * 64
        output_dim = 6
        hidden_size = 100
        # Based off reading online, it appears that Leaky ReLU could be better
        # as it eliminates vanishing gradient problem and has some sort
        # or value for negative slopes instead of 0 like it would be in
        # ReLU
        self.linear = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_size), torch.nn.LeakyReLU(), torch.nn.Linear(hidden_size, output_dim))
    def forward(self, x):
        """
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        return self.linear(x.view(x.shape[0], -1))


model_factory = {
    'linear': LinearClassifier,
    'mlp': MLPClassifier,
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
