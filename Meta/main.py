import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from Drugencoder import ChemBERTaEmbedding
from sklearn.ensemble import RandomForestClassifier
import torch.nn as nn
import torch.optim as optim
from GAN import train_gan_and_augment
from Prototype import ImprovedPrototypeNet

model_df = pd.read_csv('meta.csv', header=0)

chemBert = ChemBERTaEmbedding()
smiles_encoded = [list(chemBert(smi).detach().numpy()) for smi in model_df['smiles']]
smiles_encoded = np.array(smiles_encoded)

features = [col for col in model_df.select_dtypes(include=[np.number]).columns if col not in ['model']]
X = model_df[features].values.astype(np.float32)
y = model_df['model'].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)
n_classes = len(le.classes_)

X_combined = np.concatenate([X, smiles_encoded], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y_encoded, stratify=y_encoded, test_size=0.3, random_state=3407)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_resampled, y_train_resampled = train_gan_and_augment(X_train_scaled, y_train)

train_dataset = TensorDataset(torch.tensor(X_train_resampled), torch.tensor(y_train_resampled))
test_dataset = TensorDataset(torch.tensor(X_test_scaled), torch.tensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ImprovedPrototypeNet(input_dim=X_combined.shape[1], n_classes=n_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
criterion = nn.CrossEntropyLoss()

best_loss = float('inf')
best_accuracy = 0
patience = 20
counter = 0


n_epochs = 200
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    
    train_loss = epoch_loss / len(train_loader)
    train_accuracy = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            val_loss += criterion(logits, batch_y).item()
            
            _, predicted = torch.max(logits.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    val_loss /= len(test_loader)
    val_accuracy = correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    scheduler.step(val_loss)
    
    print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.4f} Acc: {train_accuracy:.4f} - Val Loss: {val_loss:.4f} Acc: {val_accuracy:.4f}")
    
    # 早停
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

model.load_state_dict(torch.load('best_model.pth'))