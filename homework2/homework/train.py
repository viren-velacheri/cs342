from .models import CNNClassifier, save_model
from .utils import accuracy, load_data
import torch
import torch.utils.tensorboard as tb
from tqdm.notebook import tqdm


def train(args):
    from os import path
    model = CNNClassifier()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    """
    Your code here, modify your HW1 code
    
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNClassifier()
    model.to(device)
    dataset_path = "data/train"
    dataset = load_data(dataset_path)
    global_step = 0
    loss_function = torch.nn.CrossEntropyLoss()
    lr = 0.001
    epochs = 50
    # momentum = args.momentum
    # weight_decay = args.weight_decay
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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

            # train_logger.add_scalar('loss', loss, global_step=global_step)
            global_step += 1

        model.eval()

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
