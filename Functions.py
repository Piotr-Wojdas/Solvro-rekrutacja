import numpy as np
import cv2
from PIL import Image
import torch
from hyperparameters import center_size, thresh 
from skimage.morphology import binary_closing, binary_opening

def apply_morphological_ops(x):
    # Konwersja tensora do numpy, usunięcie wymiaru kanału
    x_np = x.squeeze().numpy()
    # Zastosowanie zamknięcia binarnego do wypełnienia małych dziur
    x_closed = binary_closing(x_np)
    # Zastosowanie otwarcia binarnego do usunięcia małych szumów
    x_opened = binary_opening(x_closed)
    # Konwersja z powrotem do tensora (boolean -> float) i dodanie wymiaru kanału
    return torch.from_numpy(x_opened).float().unsqueeze(0)

def remove_grid_fft(img, adaptprog=False):
    img_np = np.array(img.convert('L')) # kolor nie ma znaczenia, konwertujemy do skali szarości
    
    # transformacja fouriera
    f = np.fft.fft2(img_np)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift) + 1)
    
    # Wykrywanie pików(lini kratki) jako lokalnych maksimów
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    local_max = cv2.dilate(magnitude_spectrum, kernel)
    peaks = (magnitude_spectrum == local_max)

    threshold = np.percentile(magnitude_spectrum, thresh)
    peaks &= (magnitude_spectrum > threshold)
    
    # Ochrona głównego kształtu (niskie częstotliwości)
    (rows, cols) = img_np.shape
    crow, ccol = rows // 2 , cols // 2
    peaks[crow-center_size:crow+center_size, ccol-center_size:ccol+center_size] = 0
    
    # usuwamy to co fft uznało za kratkę
    fshift[peaks] = 0.001
    
    # odwrotny fourier
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    img_back = cv2.normalize(img_back, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    


    return Image.fromarray(img_back)

def train_one_epoch(model, data_loader, criterion, optimizer, device):

    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        
    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

def evaluate(model, data_loader, criterion, device):

    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy, all_labels, all_preds