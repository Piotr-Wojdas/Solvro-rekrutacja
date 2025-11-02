import torch
import torch.nn as nn
import torch.optim as optim
from Functions import train_one_epoch, evaluate
from hyperparameters import num_classes, epochs, patience
from data_loader import get_dataloaders
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

# Ustawienie urządzenia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Używane urządzenie: {device}")

# --- Definicja modelu ---
# Modyfikujemy klasę, aby przyjmowała dropout_rate jako argument
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=num_classes, dropout_rate=0.5):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# --- Funkcja objective dla Optuny ---
def objective(trial):
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.6)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])

    train_loader, val_loader, _, _ = get_dataloaders()

    model = SimpleCNN(num_classes=num_classes, dropout_rate=dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=3)

    best_val_accuracy = 0.0
    for epoch in range(epochs):
        model.train()
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        model.eval()
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_accuracy


if __name__ == '__main__':
    # ==================================================================
    # ETAP 1: STROJENIE HIPERPARAMETRÓW
    # ==================================================================
    # print("="*50)
    # print("ROZPOCZYNAM ETAP 1: STROJENIE HIPERPARAMETRÓW")
    # print("="*50)
    
    # pruner = optuna.pruners.MedianPruner()
    # study = optuna.create_study(direction="maximize", pruner=pruner)
    # study.optimize(objective, n_trials=3)

    # print("\nZakończono strojenie hiperparametrów!")
    # best_params = study.best_trial.params
    # print(f"Najlepsza dokładność walidacyjna: {study.best_trial.value:.4f}")
    # print("Najlepsze hiperparametry: ", best_params)

    # # ==================================================================
    # # ETAP 2: FINALNY TRENING Z NAJLEPSZYMI PARAMETRAMI
    # # ==================================================================
    # print("\n" + "="*50)
    # print("ROZPOCZYNAM ETAP 2: FINALNY TRENING")
    # print("="*50)

    # Pobranie danych
    train_loader, val_loader, test_loader, class_names = get_dataloaders()

    # Inicjalizacja modelu z najlepszymi parametrami
    final_model = SimpleCNN(num_classes=num_classes, dropout_rate=best_params['dropout_rate']).to(device)
    criterion = nn.CrossEntropyLoss()
    
    if best_params['optimizer'] == "Adam":
        optimizer = optim.Adam(final_model.parameters(), lr=best_params['lr'])
    else:
        optimizer = optim.AdamW(final_model.parameters(), lr=best_params['lr'])
    
    best_val_accuracy = 0.0
    best_model_path = "best_model.pth"
    epochs_no_improve = 0

    # Pętla treningowa
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(final_model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(final_model, val_loader, criterion, device)
        
        print(f"Epoka [{epoch+1}/{epochs}] | "
              f"Strata treningowa: {train_loss:.4f}, Dokładność treningowa: {train_acc:.4f} | "
              f"Strata walidacyjna: {val_loss:.4f}, Dokładność walidacyjna: {val_acc:.4f}")

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(final_model.state_dict(), best_model_path)
            print(f"Zapisano nowy najlepszy model z dokładnością: {best_val_accuracy:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping, brak poprawy przez {patience} epok.")
                break

    # ==================================================================
    # ETAP 3: OSTATECZNA OCENA NA ZBIORZE TESTOWYM
    # ==================================================================
    print("\n" + "="*50)
    print("ROZPOCZYNAM ETAP 3: OSTATECZNA OCENA")
    print("="*50)
    
    # Załadowanie najlepszego modelu i ocena na zbiorze testowym
    final_model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc, all_labels, all_preds = evaluate(final_model, test_loader, criterion, device)
    print(f"\nOstateczna strata na zbiorze testowym: {test_loss:.4f}")
    print(f"Ostateczna dokładność na zbiorze testowym: {test_acc:.4f}")

    # Macierz pomyłek
    cm = confusion_matrix(all_labels, all_preds)
    print("\nMacierz pomyłek:")
    print(cm)

    # Raport klasyfikacji
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\nRaport klasyfikacji (precyzja, czułość, f1-score):")
    print(report)

    # Wizualizacja macierzy pomyłek
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Przewidziane etykiety')
    plt.ylabel('Prawdziwe etykiety')
    plt.title('Macierz pomyłek')
    plt.show()