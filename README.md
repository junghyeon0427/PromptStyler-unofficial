# PromptStyler
Unofficial code implementations of "PromptStyler: Prompt-driven Style Generation for Source-free Domain Generalization (23 ICCV)"   
  
Original paper link : https://arxiv.org/abs/2307.15199 

![Screenshot 2024-03-04 at 17 46 53](https://github.com/ai-kmu/PromptStyler/assets/77001598/c2e4a5a8-5907-4185-a9e7-8b1e25c1c242)


Reproduced by **Song Seungheon**, **Seo Junghyeon**

## Requirement

1. Clone this repository and navigate to PromptStyler folder

```
$ git clone https://github.com/ai-kmu/PromptStyler.git
$ cd PromptStyler
```

2. Install Package

```
$ conda create -n PromptStyler python=3.8 -y
$ conda activate PromptStyler
$ pip install -r requirements.txt
```

## Dataset Preparation

```
datasets
├── PACS
│   └── {domain}
│       └── {class}
├── VLCS
│   └── {domain}
│       └── {class}
├── OfficeHome
│   └── {domain}
│       └── {class}
└── DomainNet
    └── {domain}
        └── {class}
```

## (i) Prompt-driven style generation  

- `--L` is learning iteration, `--K` is number of style vectors.  
  
- After style generation, style vector in pkl format is saved in `--save_dir`.  
  
- Style vector's name format is `style_content_feature_{--model_backbone}_{L}_{K}.pkl`.  
  
```
$ python train_stylevector.py --model_backbone ViT-L/14 --data_dir /path/your/dataset --save_dir /path/your/savedir  --L 100 --K 80
```

## (ii) Training a linear classifier using diverse styles  

- `--dataset_name` is dataset name such as `PACS`, `VLCS`, `OfficeHome`, `DomainNet`.  

- After training, classifier weight is save in `--classifier_path`.

- Classifier's name format is `{--dataset_name}.pth`.  

```
$ python train_classifier.py --stylevector_path /path/your/stylevector --classifier_path /path/your/classifier --dataset_name dataset
```

## (iii) Inference using the trained classifier  

- Running the code gives you each domain accuracy.  

```
$ python test.py --model_backbone ViT-L/14 --classifier_path /path/your/classifier --data_dir /path/your/dataset --dataset_name dataset 
```
