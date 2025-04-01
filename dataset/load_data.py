# -*- coding: utf-8 -*-
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def load_data(data_dir, domain):
    data_dir = data_dir + "/" + domain
    batch_size = 64

    data_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    
    dataset = ImageFolder(data_dir, transform=data_transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if data_dir == "VLCS":
        dataset.classes = ["bird", "car", "chair", "dog", "person"]
    
    return data_loader, dataset
