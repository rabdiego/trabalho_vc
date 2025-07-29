import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import optuna
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class PreloadedImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        for label_dir in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label_dir)
            if not os.path.isdir(label_path):
                continue
            label = 1 if label_dir == "good" else 0
            for fname in os.listdir(label_path):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(label_path, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("L")  # convert to grayscale
        if self.transform:
            image = self.transform(image)
        return image, label


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

class FlexibleCNN(nn.Module):
    def __init__(self, conv_layers=2, dense_layers=1, n_filters=16, hidden_dim=128, dropout_rate=0.5):
        super().__init__()
        layers = []
        in_channels = 1
        for _ in range(conv_layers):
            layers += [
                nn.Conv2d(in_channels, n_filters, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ]
            in_channels = n_filters
            n_filters *= 2 
        self.features = nn.Sequential(*layers)

        reduced_size = 128 // (2 ** conv_layers)
        flatten_size = in_channels * reduced_size * reduced_size

        dense = [nn.Flatten()]
        for _ in range(dense_layers):
            dense.append(nn.Linear(flatten_size, hidden_dim))
            dense.append(nn.ReLU())
            dense.append(nn.Dropout(dropout_rate))
            flatten_size = hidden_dim

        dense.append(nn.Linear(flatten_size, 1))
        dense.append(nn.Sigmoid())
        self.classifier = nn.Sequential(*dense)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.squeeze(1)

full_dataset = PreloadedImageDataset('../data/', transform=transform)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

def train_model(model, loader, optimizer, criterion):
    model.train()
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.float(), labels.float())
        loss.backward()
        optimizer.step()

def evaluate_model(model, loader):
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = (outputs > 0.5).int().cpu()
            all_labels.extend(labels.cpu())
            all_preds.extend(preds)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return acc, prec, rec, f1

def objective(trial):
    batch_size = 32
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    n_filters = trial.suggest_categorical("n_filters", [4, 8, 16])
    dropout = trial.suggest_float("dropout", 0.2, 0.7)
    conv_layers = trial.suggest_int("conv_layers", 1, 4)
    dense_layers = trial.suggest_int("dense_layers", 1, 3)
    hidden_dim = trial.suggest_categorical("hidden_dim", [16, 32, 64])

    model = FlexibleCNN(conv_layers=conv_layers, dense_layers=dense_layers,
                        n_filters=n_filters, hidden_dim=hidden_dim, dropout_rate=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    for epoch in range(5):  # curtos para otimização
        train_model(model, train_loader, optimizer, criterion)
        print('epoca passou')

    acc, _, _, _ = evaluate_model(model, test_loader)
    return acc

#study = optuna.create_study(direction="maximize")
#study.optimize(objective, n_trials=20)

#best = study.best_params
model = FlexibleCNN(conv_layers=2, dense_layers=2,
                    n_filters=16, hidden_dim=64,
                    dropout_rate=0.47390094714103576).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.000509873782982352)
criterion = nn.BCELoss()
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

for epoch in range(50):
    train_model(model, train_loader, optimizer, criterion)
    acc, _, _, _ = evaluate_model(model, test_loader)
    print(f'epoca {epoch} acc {acc}')

torch.save(model.state_dict(), "cnn.pt")
acc, prec, rec, f1 = evaluate_model(model, test_loader)
print(f"Acurácia: {acc:.4f} | Precisão: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

metrics = ['Acurácia', 'Precisão', 'Recall', 'F1']
values = [acc, prec, rec, f1]
plt.bar(metrics, values)
plt.title("Desempenho Final")
plt.ylim(0, 1)
plt.savefig("metricas_final.png")
plt.show()