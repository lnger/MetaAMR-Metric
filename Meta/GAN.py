import torch
from torch.utils.data import Dataset,TensorDataset,DataLoader
import torch.nn as nn
import numpy as np


class FeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.classes = np.unique(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def get_class_samples(self, class_label):
        indices = np.where(self.y.numpy() == class_label)[0]
        return self.X[indices]

# GAN生成器
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z)

# GAN判别器
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)
    
    

# 数据增强
def train_gan_and_augment(X_train_scaled, y_train, target_samples=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    latent_dim = 32
    input_dim = X_train_scaled.shape[1]

    augmented_data = []
    augmented_labels = []
    
    for class_label in np.unique(y_train):
        class_data = X_train_scaled[y_train == class_label]
        n_samples = class_data.shape[0]
        
        if n_samples >= target_samples:
            augmented_data.append(class_data)
            augmented_labels.append(np.full(len(class_data), class_label))
            continue
            
        # 初始化GAN
        generator = Generator(latent_dim, input_dim).to(device)
        discriminator = Discriminator(input_dim).to(device)
        
        # 优化器
        g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        criterion = nn.BCELoss()
        
        # 准备真实数据
        real_data = torch.tensor(class_data, dtype=torch.float32).to(device)
        dataset = TensorDataset(real_data)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # 训练GAN
        epochs = 1000
        for epoch in range(epochs):
            for i, (real,) in enumerate(dataloader):
                batch_size = real.size(0)
                
                d_optimizer.zero_grad()
                
                real_labels = torch.ones(batch_size, 1).to(device)
                d_real_loss = criterion(discriminator(real), real_labels)

                z = torch.randn(batch_size, latent_dim).to(device)
                fake = generator(z)
                fake_labels = torch.zeros(batch_size, 1).to(device)
                d_fake_loss = criterion(discriminator(fake.detach()), fake_labels)
                
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                d_optimizer.step()

                g_optimizer.zero_grad()
                z = torch.randn(batch_size, latent_dim).to(device)
                fake = generator(z)
                g_loss = criterion(discriminator(fake), real_labels)
                g_loss.backward()
                g_optimizer.step()

        n_to_generate = target_samples - n_samples
        with torch.no_grad():
            z = torch.randn(n_to_generate, latent_dim).to(device)
            generated_data = generator(z).cpu().numpy()

        combined_data = np.vstack([class_data, generated_data])
        combined_labels = np.hstack([np.full(len(class_data), class_label), 
                                   np.full(len(generated_data), class_label)])
        
        augmented_data.append(combined_data)
        augmented_labels.append(combined_labels)

    X_resampled = np.vstack(augmented_data)
    y_resampled = np.hstack(augmented_labels)
    
    return X_resampled, y_resampled