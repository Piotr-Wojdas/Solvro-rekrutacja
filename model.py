import torch
import hyperparameters as hp
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

# Import funkcji z pliku Functions.py
from Functions import remove_grid_fft, visualize_transformations

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Używane urządzenie: {device}")

# Definicje transformacji
transform = transforms.Compose([
    transforms.Lambda(remove_grid_fft),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x > 0.5).float()),
    transforms.Normalize((0.5,), (0.5,))
])

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.file_paths = self._get_file_paths()
        self.labels = self._get_labels()

    def _get_file_paths(self):
        file_paths = []
        for cls in self.classes:
            class_dir = os.path.join(self.root_dir, cls)
            for filename in os.listdir(class_dir):
                file_paths.append(os.path.join(class_dir, filename))
        return file_paths

    def _get_labels(self):
        labels = []
        for file_path in self.file_paths:
            class_name = os.path.basename(os.path.dirname(file_path))
            labels.append(self.class_to_idx[class_name])
        return labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("L") # Konwersja do skali szarości
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

if __name__ == '__main__':
    dataset_path = 'dataset'
    full_dataset = ImageDataset(root_dir=dataset_path, transform=transform)
    
    print(f"Liczba klas: {len(full_dataset.classes)}")
    print(f"Całkowita liczba obrazów: {len(full_dataset)}")

    # Podział na zbiór treningowy i testowy
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    print(f"Rozmiar zbioru treningowego: {len(train_dataset)}")
    print(f"Rozmiar zbioru testowego: {len(test_dataset)}")

    # Tworzenie DataLoaderów
    train_loader = DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=hp.batch_size, shuffle=False)

    print("DataLoadery zostały utworzone.")

    # Używamy train_dataset do wizualizacji
    visualize_transformations(train_dataset)   