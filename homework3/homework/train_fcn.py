import torch
import numpy as np

from .models import FCN, save_model
from .utils import load_dense_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix
from . import dense_transforms
import torch.utils.tensorboard as tb
from tqdm.notebook import tqdm


def train(args):
    from os import path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FCN().to(device)
    # model.to(device)
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW1 / HW2 code
    Hint: Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: If you found a good data augmentation parameters for the CNN, use them here too. Use dense_transforms
    Hint: Use the log function below to debug and visualize your model
    """
    lr = 0.001
    loss_fun = torch.nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    epochs = 20
    global_step = 0
    dataset_path = "dense_data/train"
    dataset = load_dense_data(dataset_path)
    # Train the model
    for epoch in tqdm(range(epochs)):
        # Train for an epoch
        model.train()
        for image, label in dataset:
            # Move image, label to GPU
            image, label = image.to(device), label.to(device)
            
            # Compute network output
            pred = model(image)
            
            # Compute loss
            loss_val = loss_fun(pred, label.long())
            
            # Zero gradient
            optim.zero_grad()
            # Backward
            loss_val.backward()
            # Step optim
            optim.step()
            # Logging
            # logger.add_scalar('train/loss', float(loss_val), global_step=global_step)
            global_step += 1
        
        # logger.add_scalar('train/accuracy', float(metric.accuracy), global_step=global_step)
        # logger.add_scalar('train/iou', float(metric.iou), global_step=global_step)

        # TODO: Evaluate the model
        model.eval()
    save_model(model)


def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                                  label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
