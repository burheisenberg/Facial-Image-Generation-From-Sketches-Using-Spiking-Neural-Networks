import os
from PIL import Image
import torchvision.datasets
import torchvision.transforms as transforms
import torch
import global_v as glv

test_data_path = "./data/test"

class ToyCelebADataset(torch.utils.data.Dataset):
    def __init__(self, input_dir, output_dir, transform=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.input_filenames = os.listdir(input_dir)
        self.output_filenames = os.listdir(output_dir)
        self.transform = transform

    def __len__(self):
        return min(len(self.input_filenames), len(self.output_filenames))

    def __getitem__(self, idx):
        input_img_path = os.path.join(self.input_dir, self.input_filenames[idx])
        output_img_path = os.path.join(self.output_dir, self.output_filenames[idx])
        
        input_img = Image.open(input_img_path).convert('RGB')
        output_img = Image.open(output_img_path).convert('RGB')

        if self.transform:
            input_img = self.transform(input_img)
            output_img = self.transform(output_img)

        return input_img, output_img


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
    testset = ToyCelebADataset(test_data_path, test_data_path, transform=transform_test)
    trainset = torch.utils.data.Subset(trainset, range(2000))
    #testset = torch.utils.data.Subset(testset, range(3000,3100))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return trainloader, testloader
