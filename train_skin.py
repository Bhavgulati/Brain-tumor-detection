import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from torchvision.datasets import ImageFolder

TRAIN_DIR       = './Skin_Training'
TEST_DIR        = './Skin_Testing'
MODEL_SAVE_PATH = './models/skin_resnet50_model.pt'
BATCH_SIZE      = 64
EPOCHS          = 3
LEARNING_RATE   = 0.00001
device          = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225])
])

train_dataset = ImageFolder(TRAIN_DIR, transform=transform)
test_dataset  = ImageFolder(TEST_DIR,  transform=transform)
print(f'Classes: {train_dataset.classes}')
print(f'Train: {len(train_dataset)} | Test: {len(test_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

model = resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = True
n = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(n, 512), nn.SELU(), nn.Dropout(0.4),
    nn.Linear(512, 2)
)
model = model.to(device)

optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
best_val  = 0.0

print('\nEpoch    Train Acc    Val Acc')
print('-' * 35)

for epoch in range(EPOCHS):
    model.train()
    correct = total = 0
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()
        _, pred = torch.max(out, 1)
        total   += lbls.size(0)
        correct += (pred == lbls).sum().item()
    tacc = 100 * correct / total

    model.eval()
    vc = vt = 0
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            _, pred = torch.max(model(imgs), 1)
            vt += lbls.size(0)
            vc += (pred == lbls).sum().item()
    vacc = 100 * vc / vt

    if vacc > best_val:
        best_val = vacc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        saved = '  ✅ saved'
    else:
        saved = ''

    print(f'{epoch+1:<9}{tacc:<13.2f}{vacc:.2f}%{saved}')

print(f'\nBest Val Accuracy: {best_val:.2f}%')
print(f'Model saved to: {MODEL_SAVE_PATH}')
print('Skin cancer model ready — restart app.py')