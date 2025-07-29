import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from collections import Counter

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data
df = pd.read_csv(r"/kaggle/input/5kpolymers/pretrain_5k.csv")
df.columns = ['smiles']
texts = df['smiles'].tolist()
max_len = 179

# Create vocabulary
all_chars = set(''.join(texts))
char_to_idx = {char: i+2 for i, char in enumerate(all_chars)}
char_to_idx['<PAD>'] = 0
char_to_idx['<UNK>'] = 1
vocab_size = len(char_to_idx)

# Convert texts to sequences
sequences = []
for text in texts:
    seq = [char_to_idx.get(char, 1) for char in text]
    if len(seq) < max_len:
        seq.extend([0] * (max_len - len(seq)))
    else:
        seq = seq[:max_len]
    sequences.append(seq)
sequences = np.array(sequences)

# Dataset
class SMILESDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = torch.LongTensor(sequences)
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return self.sequences[idx], self.sequences[idx]

# Model
class Autoencoder(nn.Module):
    def __init__(self, vocab_size, max_len, embed_dim=32, latent_dim=64):
        super().__init__()
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.LSTM(embed_dim, latent_dim, batch_first=True)
        self.decoder = nn.LSTM(latent_dim, embed_dim, batch_first=True)
        self.output = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.encoder(x)
        h = h.squeeze(0).unsqueeze(1).repeat(1, self.max_len, 1)
        x, _ = self.decoder(h)
        return self.output(x)

# Training setup
dataset = SMILESDataset(sequences)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
model = Autoencoder(vocab_size, max_len).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters())

# Training
for epoch in range(100):
    total_loss = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}')

# Save
torch.save(model.state_dict(), 'autoencoder_model.pth')
print("Training complete!")