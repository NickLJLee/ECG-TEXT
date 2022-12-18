def freeze_text_encoder(model):
    for param in model.text_encoder.parameters():
        param.requires_grad = False

# template 
from torch.utils.data.dataloader import DataLoader
from utils_dataset import MIMIC_Text_dataset
import torch
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import os
from transformers import AutoModel, AutoTokenizer

# set url of your text encoder, you can find on huggingface
# some pre-trained model
# BioBert: https://huggingface.co/dmis-lab/biobert-base-cased-v1.2
# BioMedVLP: https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-general
# ClinicalBERT: https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT

url = "microsoft/BiomedVLP-CXR-BERT-general"
model = AutoModel.from_pretrained(url, trust_remote_code=True, revision='main')
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(url, trust_remote_code=True, revision='main')

# write your customized dataloader

for data in text_loader:
    text = data['text']
    # get text embedding
    text_embed = model(text)

# CLIP loss
def clip_loss(self, x, y, prior=None):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        sim = torch.einsum('i d, j d -> i j', x, y) * 1 

        labels = torch.arange(x.shape[0]).to(self.device)
        labels = torch.nn.functional.one_hot(labels, num_classes=-1).to(x.dtype)
        if prior is not None:
            prior = torch.corrcoef(prior)
            prior[prior<0] = 0
            prior.fill_diagonal_(0)
            prior = 1 - torch.exp(-0.2*prior)
            prior = prior.to(x.dtype)
            
            labels += prior

        loss_t = F.cross_entropy(sim, labels) 
        loss_i = F.cross_entropy(sim.T, labels) 
        return (loss_t + loss_i) / 2. 

# Covariance loss from BarlowTwins
def covar_loss(self, img_embed, text_embed):
        def off_diagonal(x):
            # return a flattened view of the off-diagonal elements of a square matrix
            n, m = x.shape
            assert n == m
            return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
        logits = torch.mm(img_embed.T, text_embed).to(self.device)

        logits.div_(self.train_batch_size)
        on_diag = torch.diagonal(logits).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(logits).pow_(2).sum()
        loss = on_diag +  0.0051*off_diag
        return loss/2