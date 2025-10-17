# SRC/data_loader.py (Versión PyTorch Corregida)
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from SRC.config import CIFAR10_LABELS, BATCH_SIZE

# Transformaciones: Normalización
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def load_and_preprocess_data():
    """
    Carga el dataset CIFAR-10 y lo prepara con DataLoaders de PyTorch.
    """
    print("Cargando y preprocesando el dataset CIFAR-10 con PyTorch...")
    
    DOWNLOAD_ROOT = './' # Descarga en la carpeta raíz
    
    # Descargar y cargar los datos
    trainset = torchvision.datasets.CIFAR10(root=DOWNLOAD_ROOT, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=DOWNLOAD_ROOT, train=False, download=True, transform=transform)

    # Dividir el training set en training y validation
    train_size = int(0.9 * len(trainset))
    val_size = len(trainset) - train_size
    train_data, val_data = torch.utils.data.random_split(trainset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    # Crear DataLoaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Tamaño del set de entrenamiento: {len(train_data)}")
    print(f"Tamaño del set de validación: {len(val_data)}")
    print(f"Tamaño del set de prueba: {len(testset)}")
    
    return train_loader, val_loader, test_loader, trainset

def visualize_cifar10_samples(trainset, num_samples=5):
    """Visualiza algunas imágenes del dataset (inversión de la normalización)."""
    import matplotlib.pyplot as plt
    
    print("\nVisualizando algunas imágenes de CIFAR-10...")
    plt.figure(figsize=(10, 2))
    
    for i in range(num_samples):
        ax = plt.subplot(1, num_samples, i + 1)
        
        img_tensor, label_index = trainset[i]
        
        # Invertir la normalización
        mean = torch.tensor((0.5, 0.5, 0.5)).view(3, 1, 1)
        std = torch.tensor((0.5, 0.5, 0.5)).view(3, 1, 1)
        img_tensor = img_tensor * std + mean
        
        np_img = img_tensor.numpy().transpose((1, 2, 0))
        class_name = CIFAR10_LABELS[label_index]
        
        plt.imshow(np.clip(np_img, 0, 1))
        plt.title(class_name, fontsize=8)
        plt.axis("off")
    
    if not os.path.exists("SALIDA"): os.makedirs("SALIDA")
    plt.tight_layout()
    plt.savefig("SALIDA/cifar10_samples.png")
    plt.close()
    print("Muestra de imágenes guardada en SALIDA/cifar10_samples.png")