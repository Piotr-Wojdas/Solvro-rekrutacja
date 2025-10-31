import torch
import hyperparameters as hp
from torchvision import datasets, transforms
from torch.utils.data import DataLoader





# DO SPRAWDZENIA TA FUNKCJA

def create_dataloader(data_dir, batch_size=32, img_size=(64, 64)):
    # Definicja transformacji, które zostaną zastosowane do każdego obrazu
    transform = transforms.Compose([
        # Konwersja do skali szarości - kolor prawdopodobnie nie jest istotny
        transforms.Grayscale(num_output_channels=1),
        # Zmiana rozmiaru wszystkich obrazów do tego samego wymiaru
        transforms.Resize(img_size),
        # Konwersja obrazu do tensora PyTorch (wartości pikseli skalowane do [0, 1])
        transforms.ToTensor(),
        # Normalizacja tensora do zakresu [-1, 1] co często pomaga w treningu
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Wczytanie zbioru danych za pomocą ImageFolder
    # Automatycznie znajduje klasy na podstawie nazw podfolderów
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Stworzenie DataLoader'a, który będzie dostarczał paczki danych
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,  # Mieszanie danych jest ważne podczas treningu
        num_workers=2  # Użycie dodatkowych procesów do ładowania danych
    )

    class_to_idx = dataset.class_to_idx
    print(f"Znaleziono {len(dataset)} obrazów w {len(class_to_idx)} klasach.")
    print("Mapowanie klas:", class_to_idx)

    return dataloader, class_to_idx

if __name__ == '__main__':
    # Przykład użycia
    DATASET_PATH = 'dataset'
    BATCH_SIZE = 64
    
    try:
        train_loader, class_map = create_dataloader(DATASET_PATH, BATCH_SIZE)

        # Pętla przez jedną paczkę danych, aby zobaczyć wynik
        images, labels = next(iter(train_loader))

        print(f"\nRozmiar paczki obrazów: {images.shape}") # [batch_size, channels, height, width]
        print(f"Rozmiar paczki etykiet: {labels.shape}")
        print(f"Przykładowe etykiety: {labels[:5]}")

    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono folderu '{DATASET_PATH}'. Upewnij się, że istnieje i zawiera podfoldery z klasami.")
    except Exception as e:
        print(f"Wystąpił błąd: {e}")
