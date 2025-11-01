import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from Functions import remove_grid_fft
from hyperparameters import img_height, img_width, num_classes, dataset_dir
from model import SimpleCNN  
from data_loader import SymbolDataset 

# Ścieżki
image_path = r"C:\Users\rondo\Desktop\cccc.png"
model_path = "best_model.pth"

# Transformacje
transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.Lambda(remove_grid_fft),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x > 0.5).float()),
    transforms.Normalize((0.5,), (0.5,))
])

# pobieramy nazwy klas
temp_dataset = SymbolDataset(root_dir=dataset_dir)
class_names = temp_dataset.classes


def classify_image(image_path, model_path):
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=num_classes).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() 

    img = Image.open(image_path).convert("L")
    transformed_img = transform(img)
    
    image_batch = transformed_img.unsqueeze(0).to(device)

    # Predykcja
    with torch.no_grad():
        output = model(image_batch)
        _, predicted_idx = torch.max(output, 1)
        predicted_class = class_names[predicted_idx.item()]

    # wynik 
    print(f"\nModel przewidział klasę: '{predicted_class}'")


if __name__ == '__main__':
    classify_image(image_path, model_path)

