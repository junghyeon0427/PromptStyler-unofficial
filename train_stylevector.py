# -*- coding: utf-8 -*-
import argparse

import clip
from clip.simple_tokenizer import *

# +
# import custom CLIP
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
from dataset.dataset import *
from dataset.load_data import *
from utils.modules import TextEncoder

# +
# fix seed
torch.manual_seed(427)
torch.cuda.manual_seed(427)
torch.cuda.manual_seed_all(427)

np.random.seed(427)


# -

def save_stylevector(args, feature_list):
    # style_vector save
    model_backbone = args.model_backbone.replace("/", "_")
    save_path = os.path.join(args.save_dir, f"style_content_feature_{model_backbone}_{args.L}_{args.K}.pkl")
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    with open(save_path, "wb") as f:
        pickle.dump(feature_list, f)
        
def train(args):
    model_backbone = args.model_backbone
    
    data_dir = args.data_dir 
    
    device = 'cuda'
    model, preprocess = clip.load(model_backbone, device=device)

    text_encoder = TextEncoder(model)
    tokenizer = SimpleTokenizer()
    
    # domain description
    domain_name = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    _, dataset = load_data(data_dir, domain_name[0])
    class_name = dataset.classes

    name_lens = [len(tokenizer.encode(name)) for name in class_name]
    
    L = args.L
    K = args.K
    
    D = model.visual.output_dim

    style_content_feature_list = []
    style_feature_list = []

    # number of S*
    n_ctx = 1

    n_cls = len(class_name)

    Lstyle = 0
    Lcontent = 0

    contrastive_loss = InfoNCE()
    
    for j in range(K):
        ctx_vectors = torch.empty(1, n_ctx, D)
        nn.init.normal_(ctx_vectors, std=0.02)
        # ctx_vectors = ctx_vectors.repeat(7, 1, 1)
        prompt_prefix = " ".join(["X"] * n_ctx)

        # optimize vector
        ctx = nn.Parameter(ctx_vectors)

        # optimizer & scheduler 
        optimizer = torch.optim.SGD([ctx], lr=0.002, momentum=0.9)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

        ctx_half1 = 'a'
        ctx_half2 = 'style of a'

        classnames = [name.replace("_", " ") for name in class_name]

        for x in range(L):
            # "a X style of a [class]" -> consistency loss에 사용
            prompts = [ctx_half1 + " " + prompt_prefix + " " + ctx_half2 + " " + name for name in classnames]
            # "a X style of a [class]"라는 prompt를 tokenize
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
            # "a X style of a [class]"라는 prompt에 대한 embedding
            embedding = model.token_embedding(tokenized_prompts)

            # "a X style of a" -> style loss에 사용
            prompts2 = [ctx_half1 + " " + prompt_prefix + " " + ctx_half2 + " "]
            # "a X style of a"라는 prompt를 tokenize
            tokenized_prompts2 = clip.tokenize(prompts2).to(device)
            # "a X style of a"라는 prompt에 대한 embedding
            embedding2 = model.token_embedding(tokenized_prompts2)


            # "[class]"
            prompts3 = classnames
            # "[class]"라는 prompt를 tokenize
            tokenized_prompts3 = clip.tokenize(prompts3).to(device)
            # "[class]"라는 prompt에 대한 embedding
            embedding3 = model.token_embedding(tokenized_prompts3)

            # "a X style of a"라는 prompt를 tokenize
            # prompts2 = [ctx_half1 + " " + prompt_prefix + " " + ctx_half2]

            # "a X style of a [class]"
            # "SOS a"
            prefix = embedding[:, :2, :].to(device)
            # "style of a [class] EOS"   
            suffix = embedding[:, 2 + n_ctx :, :].to(device)

            # "a X style of a"
            # "SOS a"
            prefix2 = embedding2[:, :2, :].to(device)
            # "style of a EOS"   
            suffix2 = embedding2[:, 2 + n_ctx :, :].to(device)

            prompts = []
            prompts2 = []

            # "a X style of a [class]"라는 임베딩 값을 "a S* style of a [class]"로 변경
            for i in range(n_cls):
                name_len = name_lens[i]
                # "[SOS] a"
                prefix_i = prefix[i : i + 1, :, :].to(device)
                # "style of a [class] [EOS]"
                suffix_i = suffix[i : i + 1, :, :].to(device)
                # repeat 없이 (1, 1, 512)로 만든 벡터를 계속 사용
                ctx_i = ctx[:, :, :].to(device)
                # "[SOS] a" + "S*" + "style of a [class] [EOS]"
                prompt = torch.cat([prefix_i, ctx_i, suffix_i], dim=1)
                prompts.append(prompt)

            # style_context_vector의 크기는 (K, class_num, dimension)
            prompts = torch.cat(prompts, dim=0)

            # "a X style of a"라는 임베딩 값을 "a S* style of a"로 변경
            ctx2_i = ctx[:, :, :].to(device)
            prefix2_i = prefix2[:, :, :].to(device)
            suffix2_i = suffix2[:, :, :].to(device)

            prompts2 = torch.cat((prefix2_i, ctx_i, suffix2_i), dim=1).half()

            # style_feature : "a S* style of a"
            style_feature = text_encoder(prompts2.half(), tokenized_prompts2.long())
            # style_feature : "a S* style of a [class]"
            style_content_feature = text_encoder(prompts.half(), tokenized_prompts.long())
            # content_feature : "[class]"
            content_feature = text_encoder(embedding3.half(), tokenized_prompts3.long())

            # Lstyle loss
            if j != 0:
                Lstyle = style_diversity_loss(style_feature_list, style_feature, j)

            Lcontent = contrastive_loss(style_content_feature, content_feature) / n_cls

            loss = Lcontent + Lstyle

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"{j+1} vector loss : {loss.item():.4f}")

        style_feature_list.append(style_feature.detach())
        style_content_feature_list.append(style_content_feature.detach())

    style_content_feature_list = torch.stack(style_content_feature_list, dim=0)
    style_content_feature_list = style_content_feature_list.squeeze(1)

    feature_list = style_content_feature_list.detach().to(device)
    
    save_stylevector(args, feature_list)


def main(args):
    train(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_backbone", type=str, default="ViT-L/14", help="CLIP model backbone")
    parser.add_argument("--data_dir", type=str, help="path to dataset")
    parser.add_argument("--save_dir", type=str, default="./stylevector", help="save_stylevector_dir")
    parser.add_argument("--L", type=int, default="100", help="training iteration")
    parser.add_argument("--K", type=int, default="80", help="number of style vectors")
    
    args = parser.parse_args()
    main(args)
