import os
from PIL import Image
import torchvision.datasets
import torchvision.transforms as transforms
import torch
import global_v as glv

class ToyCelebADataset(torch.utils.data.Dataset):
    def __init__(self, input_dir, features_file, transform=None):
        self.input_dir = input_dir
        self.features_file = features_file
        self.transform = transform

        # Get list of image filenames
        self.image_filenames = os.listdir(input_dir)
        self.image_filenames.sort()

        # Load features from the features file
        self.features = self._load_features()

    def _load_features(self):
        features = {}
        with open(self.features_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                image_name = parts[0]
                features[image_name] = list(map(int, parts[1:]))
        return features

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_name = self.image_filenames[idx]
        image_path = os.path.join(self.input_dir, image_name)
        image = Image.open(image_path).convert('RGB')

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        # Get corresponding features
        features = self.features[image_name]

        # Convert features to PyTorch tensor
        features = torch.tensor(features, dtype=torch.float32)
    
        return image, features


class SetRange(object):
    def __call__(self, X):
        return 2 * X - 1.

def load_toyceleba(input_data_path,output_data_path):
    print("loading ToyCelebA")
    if not os.path.exists(input_data_path):
        os.mkdir(input_data_path)
    if not os.path.exists(output_data_path):
        os.mkdir(output_data_path)

    batch_size = glv.network_config['batch_size']
    input_size = glv.network_config['input_size']

    transform_train = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        #transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        #transforms.RandomRotation(degrees=15),  # Randomly rotate the image by up to 15 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust brightness, contrast, saturation, and hue
        transforms.ToTensor(),
        SetRange()
    ])

    transform_test = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange()
    ])
    trainset = ToyCelebADataset(input_data_path, output_data_path, transform=transform_train)
    testset = ToyCelebADataset(input_data_path, output_data_path, transform=transform_test)
    #trainset = torch.utils.data.Subset(trainset, range(200))
    #testset = torch.utils.data.Subset(testset, range(10))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return trainloader, testloader
