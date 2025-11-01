import torch
import argparse
from PIL import Image
from torchvision import transforms
import os

# Importy z innych plików w projekcie
from model import SimpleCNN
from hyperparameters import img_height, img_width, num_classes, dataset_dir
from Functions import remove_grid_fft

def classify_image(image_path, model, transform, device, class_names):
    """
    Klasyfikuje pojedynczy obraz.

    Args:
        image_path (str): Ścieżka do pliku obrazu.
        model (torch.nn.Module): Wytrenowany model.
        transform (transforms.Compose): Transformacje do zastosowania na obrazie.
        device (torch.device): Urządzenie (CPU lub CUDA).
        class_names (list): Lista nazw klas.

    Returns:
        str: Nazwa przewidzianej klasy.
        float: Pewność przewidywania (softmax).
    """
    try:
        # Wczytanie obrazu i konwersja do skali szarości
        image = Image.open(image_path).convert('L')
    except FileNotFoundError:
        return f"Błąd: Plik nie został znaleziony pod ścieżką: {image_path}", 0.0
    except Exception as e:
        return f"Błąd podczas wczytywania obrazu: {e}", 0.0

    # Zastosowanie transformacji
    image_tensor = transform(image).unsqueeze(0)  # Dodanie wymiaru batcha
    image_tensor = image_tensor.to(device)

    # Klasyfikacja
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        predicted_class = class_names[predicted_idx.item()]

    return predicted_class, confidence.item()

if __name__ == '__main__':
    # --- ŚCIEŻKA DO OBRAZU ---
    # Wklej tutaj ścieżkę do obrazu, który chcesz sklasyfikować
    image_path_to_classify = r"C:\Users\rondo\Desktop\PROJEKTY\Solvro-rekrutacja\dataset\spiral\stamp05.jpg"    

    # --- Konfiguracja ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'best_model.pth'

    # --- Ładowanie modelu ---
    model = SimpleCNN(num_classes=num_classes).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku modelu '{model_path}'. Upewnij się, że model został wytrenowany i zapisany.")
        exit()
    except Exception as e:
        print(f"Błąd podczas ładowania modelu: {e}")
        exit()
        
    print(f"Model '{model_path}' został pomyślnie załadowany. Używane urządzenie: {device}")

    # --- Definicja transformacji (takie same jak dla zbioru walidacyjnego/testowego) ---
    transform = transforms.Compose([
        transforms.Lambda(lambda img: remove_grid_fft(img)),
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # --- Pobranie nazw klas z folderów w `dataset_dir` ---
    try:
        class_names = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
        if len(class_names) != num_classes:
            print(f"Ostrzeżenie: Znaleziono {len(class_names)} folderów klas, ale `num_classes` jest ustawione na {num_classes}.")
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono folderu z danymi '{dataset_dir}'. Nazwy klas nie mogą zostać wczytane.")
        exit()


    # --- Klasyfikacja obrazu ---
    predicted_class, confidence = classify_image(image_path_to_classify, model, transform, device, class_names)

    # --- Wyświetlenie wyniku ---
    if "Błąd" in predicted_class:
        print(predicted_class)
    else:
        print(f"\nPrzewidziana klasa: '{predicted_class}'")
        print(f"Pewność: {confidence:.2%}")
