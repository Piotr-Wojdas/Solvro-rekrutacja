import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import random
import os

def remove_grid_fft(img):
    """
    Usuwa siatkę z obrazu (PIL Image) za pomocą transformacji Fouriera.
    """
    if not isinstance(img, Image.Image):
        raise TypeError("img should be a PIL Image")

    # Konwersja do numpy array w skali szarości
    img_np = np.array(img.convert('L'))
    
    # Transformacja Fouriera
    f = np.fft.fft2(img_np)
    fshift = np.fft.fftshift(f)
    
    # Obliczenie spektrum mocy
    magnitude_spectrum = np.abs(fshift)
    
    # Utworzenie maski do usunięcia pików od siatki
    (rows, cols) = img_np.shape
    crow, ccol = rows // 2 , cols // 2
    
    # Heurystyczne wykrywanie pików - próg 99.8%
    threshold = np.percentile(magnitude_spectrum, 99.8)
    peaks = magnitude_spectrum > threshold
    
    # Chronimy centralny obszar spektrum (niskie częstotliwości, czyli główny kształt)
    center_size = 20
    peaks[crow-center_size:crow+center_size, ccol-center_size:ccol+center_size] = 0
    
    # Zastosowanie maski - "wymazanie" pików przez zastąpienie ich wartością bliską zeru
    fshift[peaks] = 0.001
    
    # Odwrotna transformacja Fouriera
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    # Normalizacja i konwersja z powrotem do obrazu PIL
    img_back = cv2.normalize(img_back, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    return Image.fromarray(img_back)

def visualize_transformations(dataset, num_images=10):
    """
    Wizualizuje efekt transformacji na losowych obrazkach ze zbioru danych.
    Działa zarówno z obiektem Dataset, jak i Subset (wynik random_split).
    """
    fig, axes = plt.subplots(num_images, 3, figsize=(12, num_images * 2.5))
    fig.suptitle('Efekt transformacji na losowych obrazkach', fontsize=16)
    
    # Sprawdzenie, czy dataset jest podzbiorem (Subset) czy pełnym zbiorem (Dataset)
    is_subset = hasattr(dataset, 'dataset')
    
    random_indices = random.sample(range(len(dataset)), num_images)
    
    for i, idx in enumerate(random_indices):
        # Pobranie danych w zależności od typu datasetu
        if is_subset:
            # Mamy do czynienia z torch.utils.data.Subset
            actual_idx = dataset.indices[idx]
            source_dataset = dataset.dataset
            img_path = source_dataset.file_paths[actual_idx]
            transformed_img, label = dataset[idx]
        else:
            # Mamy do czynienia ze zwykłym Datasetem
            img_path = dataset.file_paths[idx]
            transformed_img, label = dataset[idx]

        original_img = Image.open(img_path).convert("L")
        
        # Wyświetlenie oryginalnego obrazu
        ax = axes[i, 0]
        ax.imshow(original_img, cmap='gray')
        ax.set_title(f'Klasa: {dataset.dataset.classes[label]}')
        ax.axis('off')

        # Wyświetlenie strzałki
        ax = axes[i, 1]
        ax.text(0.5, 0.5, '→', ha='center', va='center', fontsize=20)
        ax.axis('off')
        
        # Wyświetlenie przetransformowanego obrazu
        ax = axes[i, 2]
        # Denormalizacja i zmiana wymiarów do wyświetlenia
        transformed_img_display = transformed_img.numpy().squeeze()
        if transformed_img_display.min() < 0: # Jeśli była normalizacja
            transformed_img_display = transformed_img_display / 2 + 0.5
        
        ax.imshow(transformed_img_display, cmap='gray')
        ax.set_title('Po transformacji')
        ax.axis('off')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()