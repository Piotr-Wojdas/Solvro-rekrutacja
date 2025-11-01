import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from Functions import train_one_epoch, evaluate
from hyperparameters import img_height, img_width, num_classes, epochs, learning_rate, patience
from data_loader import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Używane urządzenie: {device}")

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=num_classes):
        super(SimpleCNN, self).__init__()
        # Wejście: (batch_size, 1, 64, 64)
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # Po conv1: (batch_size, 32, 64, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Po pool1: (batch_size, 32, 32, 32)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Po conv2: (batch_size, 64, 32, 32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Po pool2: (batch_size, 64, 16, 16)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # Po conv3: (batch_size, 128, 16, 16)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Po pool3: (batch_size, 128, 8, 8)
        
        # Spłaszczenie tensora. Wymiar jest obliczany na podstawie wyjścia z ostatniej warstwy MaxPool
        # 128 (kanały) * 8 (wysokość) * 8 (szerokość)
        self.fc1 = nn.Linear(128 * (img_height // 8) * (img_width // 8), 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        # Spłaszczenie tensora przed warstwami gęstymi
        x = x.view(-1, 128 * (img_height // 8) * (img_width // 8))
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        # W PyTorch, dla CrossEntropyLoss, nie ma potrzeby stosowania Softmax na końcu
        return x





# Tworzy instancję modelu i wyświetla jego architekturę
model = SimpleCNN()

if __name__ == '__main__':
    # --- Inicjalizacja ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używane urządzenie: {device}") 
    train_loader, val_loader, test_loader = get_dataloaders()

    # --- Inicjalizacja modelu, funkcji straty i optymalizatora ---
    model = SimpleCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_accuracy = 0.0
    best_model_path = "best_model.pth"
    epochs_no_improve = 0

    # --- Główna pętla treningowa ---
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoka [{epoch+1}/{epochs}] | "
              f"Strata treningowa: {train_loss:.4f}, Dokładność treningowa: {train_acc:.4f} | "
              f"Strata walidacyjna: {val_loss:.4f}, Dokładność walidacyjna: {val_acc:.4f}")

        # Zapisywanie najlepszego modelu i logika early stopping
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Zapisano nowy najlepszy model z dokładnością: {best_val_accuracy:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience + 10: # niech zrobi co najmniej 10 epok za każdym razem
                print(f"\nEarly stopping, brak poprawy przez {patience} epok.")
                break

    # --- Ostateczne testowanie ---
    print("\nTestowanie najlepszego modelu na zbiorze testowym...")
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Ostateczna strata na zbiorze testowym: {test_loss:.4f}")
    print(f"Ostateczna dokładność na zbiorze testowym: {test_acc:.4f}")












