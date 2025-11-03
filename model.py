import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Functions import train_one_epoch, evaluate
from hyperparameters import  num_classes, epochs, learning_rate, patience
from data_loader import get_dataloaders
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Używane urządzenie: {device}")

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Wyjście: (batch_size, 32, 65, 80)
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Wyjście: (batch_size, 64, 32, 40)
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Wyjście: (batch_size, 128, 16, 20)
        )
        
        
        # rozmiar obrazów na przestrzeni tworzenia kodu się zmieniał, adaptive pool zawsze sprowadza do tego samego rozmiaru
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7)) 
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.classifier(x)
        return x

model = SimpleCNN()

if __name__ == '__main__':
    train_loader, val_loader, test_loader, class_names = get_dataloaders()

    # --- Inicjalizacja modelu, funkcji straty i optymalizatora ---
    model = SimpleCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)    
    best_val_accuracy = 0.0
    best_model_path = "best_model.pth"
    epochs_no_improve = 0

    # pętla treningowa
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _= evaluate(model, val_loader, criterion, device)
        
        # print(f"Epoka [{epoch+1}/{epochs}] | "
        #       f"Strata treningowa: {train_loss:.4f}, Dokładność treningowa: {train_acc:.4f} | "
        #       f"Strata walidacyjna: {val_loss:.4f}, Dokładność walidacyjna: {val_acc:.4f}")

        # early stopping na najlepszym modelu
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Epoka [{epoch+1}/{epochs}] | "
              f"Strata treningowa: {train_loss:.4f}, Dokładność treningowa: {train_acc:.4f} | "
              f"Strata walidacyjna: {val_loss:.4f}, Dokładność walidacyjna: {val_acc:.4f}")
            print(f"Zapisano nowy najlepszy model z dokładnością: {best_val_accuracy:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience and epoch > 30: # niech zrobi co najmniej 30 epok za każdym razem
                print(f"\nEarly stopping, brak poprawy przez {patience} epok.")
                break

    
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc, all_labels, all_preds = evaluate(model, test_loader, criterion, device)
    print(f"\nOstateczna strata na zbiorze testowym: {test_loss:.4f}")
    print(f"Ostateczna dokładność na zbiorze testowym: {test_acc:.4f}")

    # --- Obliczanie i wyświetlanie macierzy pomyłek ---
    cm = confusion_matrix(all_labels, all_preds)
    print("\nMacierz pomyłek:")
    print(cm)

    # --- Obliczanie i wyświetlanie raportu klasyfikacji (w tym precyzji) ---
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\nRaport klasyfikacji (precyzja, czułość, f1-score):")
    print(report)

    # --- Wizualizacja macierzy pomyłek ---
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Przewidziane etykiety')
    plt.ylabel('Prawdziwe etykiety')
    plt.title('Macierz pomyłek')
    plt.show()












