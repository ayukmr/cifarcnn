import os
import sys
import pickle

import numpy as np
from PIL import Image

import torch
from torch import nn

from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

EPOCHS = 20
BATCH_SIZE = 32
N_CLASSES = 100

device = torch.device('mps')

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.c1 = nn.Conv2d(3,   64,  kernel_size=4, stride=2, padding=1)
        self.c2 = nn.Conv2d(64,  128, kernel_size=4, stride=2, padding=1)
        self.c3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.f  = nn.Flatten()
        self.d1 = nn.Linear(256 * 4 * 4, N_CLASSES)

        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.relu(self.c1(x))
        x = self.relu(self.c2(x))
        x = self.relu(self.c3(x))

        x = self.f(x)
        x = self.d1(x)

        return x

def unpickle(file):
    with open(f'data/{file}.pkl', 'rb') as file:
        return pickle.load(file, encoding='bytes')

def load_batches():
    data = np.empty((0, 3072))
    labels = []

    for n in range(1, 6):
        batch = unpickle(f'batch_{n}')

        data = np.concat((data, batch[b'data']))
        labels += batch[b'labels']

    return (data, labels)

def load_data():
    data, labels = load_batches()

    images = np.array(data).astype(np.float32) / 255
    images = images.reshape(-1, 3, 32, 32)

    labels = np.array(labels)

    dataset = TensorDataset(torch.tensor(images), torch.tensor(labels))
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    return loader

def load_names():
    meta = unpickle('meta')
    return meta[b'label_names']

def train():
    os.makedirs('models', exist_ok=True)

    model = Classifier().to(device)
    optim = Adam(model.parameters(), 1e-3)

    criterion = nn.CrossEntropyLoss()

    loader = load_data()

    for epoch in range(EPOCHS):
        print(f'epoch {epoch}')
        print('---')

        for i, (data, label) in enumerate(iter(loader)):
            out = model(data.to(device))
            loss = criterion(out, label.to(device))

            loss.backward()
            optim.step()
            optim.zero_grad()

            if i % 100 == 0:
                print(f'e: {epoch:02d}, i: {i:04d} | l: {loss:.3f}')

    torch.save(model.state_dict(), 'models/model.pth')

def inference(image):
    model = Classifier().to(device)

    model.load_state_dict(torch.load('models/model.pth'))
    model.eval()

    img = Image.open(image).convert('RGB')
    tensor = torch.from_numpy(np.array(img).astype(np.float32)).to(device) / 255

    data = tensor.permute(2, 0, 1).unsqueeze(0)

    out = model(data)
    idx = out.argmax()

    classes = load_names()

    probs, idxs = torch.topk(out, 5)

    idxs  = idxs[0]
    probs = probs[0]

    print([(classes[idxs[n]], probs[n].item()) for n in range(5)])

    return classes[idx]

def main():
    if len(sys.argv) == 2 and sys.argv[1] == 'train':
        train()
    elif len(sys.argv) == 3 and sys.argv[1] == 'eval':
        inference(sys.argv[2])
    else:
        print('error: incorrect arguments')

    return 0
