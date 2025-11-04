import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from Functions import remove_grid_fft, apply_morphological_ops
from hyperparameters import img_height, img_width, num_classes, dataset_dir
from og_model import SimpleCNN
from data_loader import SymbolDataset

# Ścieżki
model_path = "THE_BEST_model.pth"
new_images_dir = "new_images"

# Transformacje
transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.Lambda(remove_grid_fft),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x > 0.5).float()),
    transforms.Normalize((0.5,), (0.5,))
])

# Pobieramy nazwy klas
try:
    temp_dataset = SymbolDataset(root_dir=dataset_dir)
    class_names = temp_dataset.classes
except Exception:
    class_names = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])


def classify_and_display_images(images_dir, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)) 
    model.eval()


    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

   

    for image_name in image_files:
        image_path = os.path.join(images_dir, image_name)
        try:
            # Otwieranie i transformacja obrazu
            img_for_display = Image.open(image_path)
            img_for_model = img_for_display.convert("L")
            transformed_img = transform(img_for_model)
            image_batch = transformed_img.unsqueeze(0).to(device)

            # Predykcja
            with torch.no_grad():
                output = model(image_batch)
                _, predicted_idx = torch.max(output, 1)
                predicted_class = class_names[predicted_idx.item()]

            # Wyświetlanie wyniku
            plt.figure()
            plt.imshow(img_for_display,cmap='gray')
            plt.title(f"Przewidziana klasa: {predicted_class}")
            plt.axis('off')
            plt.show()
            
            print(f"Plik: '{image_name}' -> Przewidziana klasa: '{predicted_class}'")

        except Exception as e:
            print(f"Nie udało się przetworzyć pliku {image_name}. Błąd: {e}")


if __name__ == '__main__':
    classify_and_display_images(new_images_dir, model_path)

