from .models import ClassificationLoss, model_factory, save_model
from .utils import accuracy, load_data
import tempfile
import torch.utils.tensorboard as tb
import time
from tqdm.notebook import tqdm
import torch


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_factory[args.model]()
    model.to(device)
    dataset_path = "data/train"
    dataset = load_data(dataset_path)
    log_dir = 'log_dir'
    logger = tb.SummaryWriter(log_dir + '/{}'.format(time.strftime('%H-%M-%S')))
    global_step = 0
    loss_function = ClassificationLoss()
    lr = args.lr
    epochs = args.epochs
    momentum = args.momentum
    weight_decay = args.weight_decay
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    for epoch in tqdm(range(epochs)):
        model.train()
        for x, y in dataset:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            loss = loss_function(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logger.add_scalar('loss', loss, global_step=global_step)
            global_step += 1

        model.eval()



    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    # Custom arguments here
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-epochs', type=int, default=75)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-weight_decay', type=float, default=1e-7)
    

    args = parser.parse_args()
    train(args)
