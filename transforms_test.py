from torchvision import transforms
from hyperparameters import img_height, img_width
from Functions import remove_grid_fft, apply_morphological_ops
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import binary_closing, binary_opening
import torch


image_path = r"new_images/aa.jpg"

# Funkcja do zastosowania zamknięcia morfologicznego w potoku transformacji
def apply_morphological_ops(x):
    # Konwersja tensora do numpy, usunięcie wymiaru kanału
    x_np = x.squeeze().numpy()
    # Zastosowanie zamknięcia binarnego do wypełnienia małych dziur
    x_closed = binary_closing(x_np)
    # Zastosowanie otwarcia binarnego do usunięcia małych szumów
    x_opened = binary_opening(x_closed)
    # Konwersja z powrotem do tensora (boolean -> float) i dodanie wymiaru kanału
    return torch.from_numpy(x_opened).float().unsqueeze(0)

# Transformacje
transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.Lambda(remove_grid_fft),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x > 0.7).float()),
    transforms.Lambda(apply_morphological_ops),
    transforms.Normalize((0.5,), (0.5,))
])

transform2 = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.Lambda(remove_grid_fft),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x > 0.5).float()),
    transforms.Normalize((0.5,), (0.5,))
])

# Wczytaj obraz
original_image = Image.open(image_path).convert("L")

# Zastosuj transformacje
transformed_image_tensor = transform2(original_image)

# Konwertuj tensor do numpy array do wizualizacji
transformed_image_np = transformed_image_tensor.squeeze().numpy()

# Wyświetl obrazy
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].imshow(original_image, cmap='gray')
ax[0].set_title("Oryginalny obraz")
ax[0].axis('off')

ax[1].imshow(transformed_image_np, cmap='gray')
ax[1].set_title("Przetworzony obraz")
ax[1].axis('off')

plt.show()