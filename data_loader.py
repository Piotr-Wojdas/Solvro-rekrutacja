import os
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
from hyperparameters import dataset_dir, batch_size, img_height, img_width
from Functions import remove_grid_fft, apply_morphological_ops

class SymbolDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_paths = []
        self.labels = []
        
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        for cls_name in self.classes:
            class_dir = os.path.join(root_dir, cls_name)
            for file_name in os.listdir(class_dir):
                self.file_paths.append(os.path.join(class_dir, file_name))
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_dataloaders(dataset_dir=dataset_dir, batch_size=batch_size, test_size=0.2, val_size=0.2, augment=True):
    
    val_test_transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.Lambda(remove_grid_fft),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.65).float()),
        transforms.Lambda(apply_morphological_ops),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.Lambda(remove_grid_fft),
            transforms.RandomAffine(degrees=20, translate=(0.15, 0.15), scale=(0.85, 1.15), shear=10),
            transforms.ElasticTransform(alpha=50.0, sigma=5.0),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > 0.65).float()),
            transforms.Lambda(apply_morphological_ops),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        train_transform = val_test_transform

    # Utworzenie pełnego zbioru danych
    full_dataset = SymbolDataset(root_dir=dataset_dir)
    
    # Podział na zbiór treningowo-walidacyjny i testowy
    indices = list(range(len(full_dataset)))
    labels = full_dataset.labels
    
    train_val_indices, test_indices = train_test_split(
        indices, test_size=test_size, stratify=labels, random_state=67
    )
    
    # Podział zbioru treningowo-walidacyjnego na treningowy i walidacyjny
    train_val_labels = [labels[i] for i in train_val_indices]
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=val_size, stratify=train_val_labels, random_state=42
    )

    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_test_transform
    test_dataset.dataset.transform = val_test_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    class_names = full_dataset.classes
    return train_loader, val_loader, test_loader, class_names

