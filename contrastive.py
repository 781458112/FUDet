import logging
import os
import pickle
from collections import OrderedDict
import torch.nn as nn
import math
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


class NoisyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, std_dev=0.1):
        super(NoisyModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.std_dev = std_dev

    def forward(self, x):
        # Ìí¼ÓÔëÉù
        noise = torch.randn_like(x) * self.std_dev
        x = self.linear1(x)
        x = self.relu(x)
        x = x + noise  # ½«ÔëÉùÌí¼Óµ½ÌØÕ÷ÏòÁ¿
        x = self.linear2(x)
        return x


class MLPWithBN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPWithBN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        # self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        # self.bn2 = nn.BatchNorm1d(output_size)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.bn1(x)
        # x = self.relu1(x)
        x = self.fc2(x)
        # x = self.bn2(x)
        return x

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # ¼ÆËãÍ¬ÀàÑù±¾ºÍ²»Í¬ÀàÑù±¾Ö®¼äµÄÅ·ÊÏ¾àÀë
        pos_distance = torch.sqrt(torch.sum((anchor - positive) ** 2, dim=1))
        neg_distance = torch.sqrt(torch.sum((anchor - negative) ** 2, dim=1))

        # ¼ÆËãËðÊ§
        loss = torch.relu(pos_distance - neg_distance + self.margin)

        # ·µ»ØÆ½¾ùËðÊ§
        return torch.mean(loss)