from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import csv

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path):

        """
        Your code here
        Hint: Use the python csv library to parse labels.csv
        

        WARNING: Do not perform data normalization here. 
        """

        with open(dataset_path + "/labels.csv", newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            dictReader = csv.DictReader(csvfile)
            self.image_labels = list()
            # transformation = transforms.toTensor()
            for row in reader:
                img = Image.open(dataset_path + "/" + row['file'])
                image_labels.append((transforms.toTensor(img)), LABEL_NAMES.index(row['label']))

        # raise NotImplementedError('SuperTuxDataset.__init__')

    def __len__(self):
        """
        Your code here
        """
        return len(self.image_labels)

    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """
        return self.image_labels.index(idx)


def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
