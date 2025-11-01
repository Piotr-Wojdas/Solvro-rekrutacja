import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import random
import os
import torch

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



def train_one_epoch(model, data_loader, criterion, optimizer, device):
    """
    Przeprowadza jedną epokę treningu.
    """
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Wyzerowanie gradientów
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass i optymalizacja
        loss.backward()
        optimizer.step()
        
        # Statystyki
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        
    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

def evaluate(model, data_loader, criterion, device):
    """
    Ocenia model na danym zbiorze danych (walidacyjnym lub testowym).
    """
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy