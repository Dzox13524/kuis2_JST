import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_set = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_set = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

class DeepLearner(nn.Module):
    def __init__(self):
        super(DeepLearner, self).__init__()
        self.relu = nn.ReLU()
        self.feature_low = nn.Linear(784, 512)
        self.feature_mid = nn.Linear(512, 256)
        self.feature_high = nn.Linear(256, 128)
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784) 
        x = self.relu(self.feature_low(x))
        x = self.relu(self.feature_mid(x))
        x = self.relu(self.feature_high(x))
        return self.classifier(x)

    def get_features(self, x):
        features = {}
        x = x.view(-1, 784)
        x = self.relu(self.feature_low(x));  features['Low'] = x
        x = self.relu(self.feature_mid(x));  features['Mid'] = x
        x = self.relu(self.feature_high(x)); features['High'] = x
        return features

model = DeepLearner().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
history = {'loss': [], 'acc': []}

print("--- Memulai Proses Feature Learning Otomatis ---")
for epoch in range(5):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    history['loss'].append(running_loss/len(train_loader))
    history['acc'].append(correct/total)
    print(f"Epoch {epoch+1} Selesai")

def visualize_results(model, test_loader, history):
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    with torch.no_grad():
        features = model.get_features(images)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Analisis Mekanisme DNN: Feature Learning Berjenjang (Khosyatullah Ahmad)', fontsize=16)

    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(history['loss'], 'r-o')
    ax1.set_title('A. Penurunan Loss (Optimalisasi)')

    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(np.array(history['acc'])*100, 'b-s')
    ax2.set_title('B. Peningkatan Akurasi')

    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(images[0].cpu().squeeze(), cmap='gray')
    ax3.set_title(f'C. Data Mentah: {classes[labels[0]]}')
    ax3.axis('off')

    ax4 = plt.subplot(2, 3, 4)
    means = [features['Low'].mean().item(), features['Mid'].mean().item(), features['High'].mean().item()]
    ax4.bar(['Low', 'Mid', 'High'], means, color=['green', 'orange', 'purple'])
    ax4.set_title('D. Rerata Aktivasi per Jenjang')

    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(features['High'][0].cpu().numpy(), bins=20, color='purple')
    ax5.set_title('E. Distribusi Fitur Lapisan Tinggi')

    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    ax6.set_title('F. Contoh Hasil Prediksi')
    for i in range(4):
        sub = fig.add_axes([0.7 + (i%2)*0.12, 0.1 + (i//2)*0.15, 0.1, 0.1])
        sub.imshow(images[i].cpu().squeeze(), cmap='gray')
        color = 'green' if preds[i] == labels[i] else 'red'
        sub.set_title(f'P:{classes[preds[i]]}', color=color, fontsize=8)
        sub.axis('off')

    plt.tight_layout(rect=dd .)
    plt.savefig('analisis_dnn_khosyatullah.png')
    plt.show()

visualize_results(model, test_loader, history)