# -*- coding: utf-8 -*-
# +
# python train_classifier.py --stylevector_path stylevector/style_content_feature_ViT-L_14_100_80.pkl --classifier_path models/ --dataset_name PACS 

# +
# import custom CLIP
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import numpy as np
import os
import pickle
import io

from utils.loss import *
from utils.modules import CPU_Unpickler, Classifier
from dataset.dataset import *
from dataset.load_data import *

# +
# fix seed
torch.manual_seed(427)
torch.cuda.manual_seed(427)
torch.cuda.manual_seed_all(427)

np.random.seed(427)


# -

def load_stylevector(args):
    # style_vector load
    with open(args.stylevector_path, "rb") as f:
        feature_list = CPU_Unpickler(f).load()
        
    return feature_list

def save_classifier(args, classifier):
    if not os.path.exists(args.classifier_path):
        os.makedirs(args.classifier_path)
    
    # Save classifier
    torch.save(classifier.state_dict(), os.path.join(args.classifier_path, f'{args.dataset_name}.pth'))

def train(args):
    device = 'cuda'
    feature_list = load_stylevector(args)
    K, n_cls, D = feature_list.shape
    
    feature_list = feature_list.view(K * n_cls, D).to(device)
    label = torch.arange(n_cls).repeat(K).to(device)
    
    dataset = CustomDataset(feature_list, label)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    classifier = Classifier(D, n_cls).to(dtype=torch.float32).to(device)

    optimizer = torch.optim.SGD(classifier.parameters(), lr=args.lr, momentum=args.momentum)

    criterion = nn.CrossEntropyLoss()
    
    classifier.train()

    for i in range(args.epoch):
        for data in dataloader:

            feature, label = data
            feature = feature.to(dtype=torch.float32)
            feature = feature / feature.norm(dim=1, keepdim=True)

            loss = classifier(feature, label)

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()

        print(f"{i+1} epoch loss : {loss.item()} ")
        
    save_classifier(args, classifier)


def main(args):
    train(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--stylevector_path", type=str, help="path to stylevector")
    parser.add_argument("--classifier_path", type=str, help="path to classifier")
    parser.add_argument("--dataset_name", type=str, help="dataset_name")
    parser.add_argument("--epoch", type=int, default=50, help="training epoch")
    parser.add_argument("--lr", type=int, default=0.005, help="learning rate")
    parser.add_argument("--momentum", type=int, default=0.9, help="momentum")
    
    args = parser.parse_args()
    main(args)
