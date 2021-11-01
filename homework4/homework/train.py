import torch
import numpy as np

from .models import Detector, save_model
from .utils import load_detection_data
from . import dense_transforms
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    model = Detector()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW3 code
    Hint: Use the log function below to debug and visualize your model
    """


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Detector().to(device)
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    loss = torch.nn.BCEWithLogitsLoss()
    transformation = dense_transforms.Compose([dense_transforms.ColorJitter(0.9, 0.9, 0.9, 0.1), dense_transforms.RandomHorizontalFlip(), dense_transforms.ToTensor(), dense_transforms.ToHeatmap()])
    train_data = load_detection_data('dense_data/train', num_workers=2, transform=transformation)

    global_step = 0
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        for img, label, extra_val in train_data:
            img, label = img.to(device), label.to(device)

            logit = model(img)
            loss_val = loss(logit, label)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1

        model.eval()
        print(epoch)

    save_model(model)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    args = parser.parse_args()
    train(args)
