# -*- coding: utf-8 -*-
import clip
from clip.simple_tokenizer import *

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

def evaluate_model(classifier, model, data_loader, device):
    classifier.eval()
    
    for name, param in model.named_parameters():
        if name == "logit_scale":
            logit_scale = param
            
    logit_scale = logit_scale.exp()
    
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            image_feature = model.encode_image(images)
            image_feature = image_feature / image_feature.norm(dim=1, keepdim=True)
            image_feature = logit_scale * image_feature
            image_feature = image_feature.to(dtype=torch.float32)
            
            outputs = classifier(image_feature, embed=True)

            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

        total = len(y_pred)
        correct = 0
        
        for pred, true in zip(y_pred, y_true):
            if pred == true:
                correct += 1
        
        accuracy = correct / total
        
    return accuracy


def test(args):
    device = 'cuda'
    total_acc = 0
    
    model, preprocess = clip.load(args.model_backbone, device=device)
    
    a = torch.load(args.classifier_path)
    
    n_cls, D = list(a.values())[0].shape
    
    classifier = Classifier(D, n_cls).to(dtype=torch.float32).to(device)
    classifier.load_state_dict(a)
    
    domain_name = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]
    
    for domain in domain_name:
        data_loader, _ = load_data(args.data_dir, domain)

        accuracy = evaluate_model(classifier, model, data_loader, device)
        print(f'Accuracy on the {args.dataset_name} dataset, {domain} domain: {accuracy * 100:.2f}%')
        total_acc += accuracy

    print(f'Total Accuracy {(total_acc/4) * 100:.2f}%')


def main(args):
    test(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_backbone", type=str, default="ViT-L/14", help="CLIP model backbone")
    parser.add_argument("--classifier_path", type=str, help="path to classifier_path")
    parser.add_argument("--data_dir", type=str, help="path to dataset")
    parser.add_argument("--dataset_name", type=str, help="dataset name")
        
    args = parser.parse_args()
    main(args)
