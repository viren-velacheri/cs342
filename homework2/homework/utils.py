from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import csv

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']

class SuperTuxDataset(Dataset):
    """
    WARNING: Do not perform data normalization here. 
    """
    def __init__(self, dataset_path):
        """
        Your code here
        Hint: Use your solution (or the master solution) to HW1
        """

        transformation = transforms.ToTensor()
        labels_csv_path = dataset_path + "/labels.csv"
        with open(labels_csv_path, newline='') as csvfile:
          dictReader = csv.DictReader(csvfile)
          self.image_labels = list()
          for row in dictReader:
            image_file_path = dataset_path + "/" + row['file']
            img = Image.open(image_file_path)
            self.image_labels.append((transformation(img), LABEL_NAMES.index(row['label'])))

    def __len__(self):
      return len(self.image_labels)


    def __getitem__(self, idx):
      return self.image_labels[idx]


def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
