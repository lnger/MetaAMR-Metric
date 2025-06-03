import pandas as pd
from scipy.sparse import coo_matrix
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool,GINEConv
from torch_geometric.data import Data, DataLoader
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, DataLoader, Batch
from torch import nn
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
import torch.optim as optim

import copy
import random
from collections import Counter
import Biodata
from torch_geometric.nn import GINEConv,TransformerConv

class CustomModel(torch.nn.Module):
    def __init__(self, input_dim,hidden_dim_1, hidden_dim_2, output_dim, edge_dim):
        super(CustomModel, self).__init__()
        self.conv1 = GINEConv(
            nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim_1),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim_1, hidden_dim_1)
            ),
            edge_dim=edge_dim  # 边特征
        )

        self.conv2 = GINEConv(
            nn.Sequential(
                torch.nn.Linear(hidden_dim_1, hidden_dim_2),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim_2, hidden_dim_2)
            ),
            edge_dim=edge_dim
        )

        self.conv3 = GINEConv(
            nn.Sequential(
                torch.nn.Linear(hidden_dim_2, hidden_dim_1),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim_1, hidden_dim_1)
            ),
            edge_dim=edge_dim
        )

        self.projector = nn.Linear(hidden_dim_1, 128)
        self.pool = global_mean_pool
        self.classifier = torch.nn.Linear(hidden_dim_1, output_dim)
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.leaky_relu(self.conv1(x, edge_index, edge_attr))
        x = F.leaky_relu(self.conv2(x, edge_index, edge_attr))
        x = F.leaky_relu(self.conv3(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.3, training=self.training) 
        x = self.pool(x, batch)
        logits = self.classifier(x)
        features = F.normalize(self.projector(x), dim=-1)
        return logits, features
def train(dataset, model, learning_rate=1e-5, batch_size=64, epoch_n=20, random_seed=111, val_split=0.2, weighted_sampling=True, model_name="GCN_model.pt", device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    random.seed(random_seed)
    data_list = list(range(0, len(dataset)))
    test_list = random.sample(data_list, int(len(dataset) * val_split))
    trainset = [dataset[i] for i in data_list if i not in test_list]
    testset = [dataset[i] for i in data_list if i in test_list]

    best_metrics = {
        'precision': 0,
        'epoch': -1,
        'acc': 0,
        'mcc': 0,
        'f1': 0,
        'auroc': 0,
        'auprc': 0
    }
    
    if weighted_sampling:
        label_count = Counter([int(data.y) for data in dataset])
        weights = [100/label_count[int(data.y)] for data in trainset]
        sampler = WeightedRandomSampler(weights, num_samples=len(trainset), replacement=True)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False,sampler=sampler)
    else:
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    alpha = 0.5

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # 使用AdamW

    for epoch in range(epoch_n):
        training_running_loss = 0.0
        train_acc = 0.0
        
        model.train()
        for i, batch in enumerate(train_loader):
            batch = batch.to(device)
            # label = batch.y.float()
            label = batch.y

            pred, features = model(batch)
            # print(pred)
            ce_loss = criterion(pred, label)
            total_loss = ce_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            training_running_loss += total_loss.detach().item()
            train_acc += (torch.argmax(pred, 1).flatten() == label).type(torch.float).mean().item()
            
        test_acc, test_precision, test_mcc, test_f1, test_auroc, test_auprc = evaluation(test_loader, model, device)


        if test_precision > best_metrics['precision']:
            best_metrics = {
                'epoch': epoch,
                'precision': test_precision,
                'acc': test_acc,
                'mcc': test_mcc,
                'f1': test_f1,
                'auroc': test_auroc,
                'auprc': test_auprc
            }

    print("\n=== Best Epoch (Based on Validation Precision) ===")
    print(f"Epoch {best_metrics['epoch']} Results:")
    print(f"Val Accuracy: {best_metrics['acc']:.4f}")
    print(f"Precision:    {best_metrics['precision']:.4f}")
    print(f"MCC:          {best_metrics['mcc']:.4f}")
    print(f"F1-Score:     {best_metrics['f1']:.4f}")
    print(f"AUROC:        {best_metrics['auroc']:.4f}")
    print(f"AUPRC:        {best_metrics['auprc']:.4f}")
    print("===                                           ===")

    return model

from sklearn.metrics import accuracy_score,f1_score, precision_score,matthews_corrcoef,roc_auc_score, average_precision_score
def evaluation(loader, model, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    
    for data in loader:
        with torch.no_grad():
            data = data.to(device)
            pred, _ = model(data)
            probs = torch.softmax(pred, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())  # 假设正类是第1类
            pred = pred.argmax(dim=1)
            all_labels.extend(data.y.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auroc = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)
    
    return acc, precision, mcc, f1, auroc, auprc

