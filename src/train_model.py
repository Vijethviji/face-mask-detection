import os
import torch
import torchvision
from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# ==== UPDATE THIS PATH to your actual data folder ====
data_dir = r"C:\Users\Vijeth B V\OneDrive\Desktop\abhijna\dataset"

# Check if data folder exists
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Data folder not found at {data_dir}")

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Dataset & Dataloader
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Split dataset 80-20
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model: MobileNetV2 pretrained with modified classifier
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, 2)  # 2 classes: mask, no mask
model = model.to(device)

# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 32
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")

    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item(), acc=100*correct/total)

    # Validation accuracy
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            val_outputs = model(val_inputs)
            _, val_preds = torch.max(val_outputs, 1)
            val_total += val_labels.size(0)
            val_correct += (val_preds == val_labels).sum().item()

    print(f"Validation Accuracy: {100 * val_correct / val_total:.2f}%")

# Save model
save_path = r"C:\Users\Vijeth B V\OneDrive\Desktop\abhijna\models\mask_detector_model.pth"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(model, save_path)
print(f"Model saved to {save_path}")
