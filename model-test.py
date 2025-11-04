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
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Używane urządzenie: {device}")

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            # Ostatnia warstwa konwolucyjna, do której podepniemy Grad-CAM
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4)) 
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Miejsce na przechowywanie gradientów i aktywacji
        self.gradients = None
        self.activations = None

    # Hook do przechwytywania gradientów
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x, register_hook=False):
        x = self.features(x)
        
        if register_hook:
            # Rejestrujemy hook na ostatniej warstwie konwolucyjnej
            h = x.register_hook(self.activations_hook)
            self.activations = x
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    # Metoda do pobierania gradientów
    def get_activations_gradient(self):
        return self.gradients

    # Metoda do pobierania aktywacji
    def get_activations(self):
        return self.activations

def generate_grad_cam(model, img_tensor, class_names, original_img_np, pred_class_idx):
    model.eval()
    
    # Przepuszczamy obraz przez model z włączonym hookiem
    output = model(img_tensor, register_hook=True)
    
    # Zerujemy gradienty
    model.zero_grad()
    
    # Pobieramy wynik dla przewidzianej klasy
    pred_class_score = output[:, pred_class_idx]
    pred_class_score.backward()
    
    # Pobieramy gradienty i aktywacje
    gradients = model.get_activations_gradient()
    activations = model.get_activations().detach()
    
    # Obliczamy wagi kanałów
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
        
    # Tworzymy heatmapę
    heatmap = torch.mean(activations, dim=1).squeeze().cpu()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= torch.max(heatmap)
    
    # Wizualizacja
    heatmap = cv2.resize(heatmap.numpy(), (original_img_np.shape[1], original_img_np.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Normalizacja oryginalnego obrazu do 0-255
    original_img_normalized = cv2.normalize(original_img_np, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    if len(original_img_normalized.shape) == 2:
        original_img_normalized = cv2.cvtColor(original_img_normalized, cv2.COLOR_GRAY2BGR)

    superimposed_img = heatmap * 0.4 + original_img_normalized
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    return superimposed_img, class_names[pred_class_idx]

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
        
        print(f"Epoka [{epoch+1}/{epochs}] | "
              f"Strata treningowa: {train_loss:.4f}, Dokładność treningowa: {train_acc:.4f} | "
              f"Strata walidacyjna: {val_loss:.4f}, Dokładność walidacyjna: {val_acc:.4f}")

        # early stopping na najlepszym modelu
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), best_model_path)
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

    # --- Generowanie Grad-CAM dla kilku obrazów z zestawu testowego ---
    print("\nGenerowanie wizualizacji Grad-CAM...")
    num_images_to_show = 5
    
    # Użyjemy oryginalnego datasetu bez augmentacji, aby pobrać obrazy
    _, _, test_loader_for_cam, _ = get_dataloaders(augment=False)
    
    data_iter = iter(test_loader_for_cam)
    images, labels = next(data_iter)

    for i in range(num_images_to_show):
        img_tensor = images[i].unsqueeze(0).to(device)
        original_img_np = images[i].squeeze().numpy()

        # Predykcja modelu
        output = model(img_tensor)
        _, pred_idx = torch.max(output, 1)
        
        superimposed_img, pred_class_name = generate_grad_cam(model, img_tensor, class_names, original_img_np, pred_idx.item())
        
        # Wyświetlanie
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(original_img_np, cmap='gray')
        plt.title(f"Oryginał, Prawdziwa klasa: {class_names[labels[i]]}")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Grad-CAM, Przewidziana klasa: {pred_class_name}")
        plt.axis('off')
        
        plt.show()
