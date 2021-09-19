from .models import ClassificationLoss, model_factory, save_model
from .utils import accuracy, load_data
from tqdm.notebook import tqdm


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_factory[args.model](3 * 64 * 64, 6)
    model.to(device)
    train_path = "data/train"
    valid_path = "data/valid"
    data_train = load_data(dataset_path)
    data_val = load_data(valid_path)
    lr = args.lr
    epochs = args.epochs
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.85, weight_decay=1e-7)

    for epoch in tqdm(range(epochs)):
        model.train()
        for x, y in data_train:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            loss = ClassificationLoss(output, y)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()



    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    parser.add_argument('-m', '--lr', type=float, default=0.001)
    parser.add_argument('-m', '--epochs', type=int, default=20)
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
