# SRC/model.py (PyTorch Version)
import torch
import torch.nn as nn
import torch.nn.functional as F
from SRC.config import KERNEL_SIZE, POOL_SIZE, FILTERS_1, FILTERS_2, DENSE_UNITS_1, NUM_CLASSES, INPUT_CHANNELS

class SimpleCNN(nn.Module):
    """
    Implementación de la CNN usando nn.Module de PyTorch.
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # ----------------------------------------
        # Extractor de Características
        # ----------------------------------------
        
        # Bloque Convolucional 1: Conv2D -> MaxPooling2D
        self.conv1 = nn.Conv2d(
            in_channels=INPUT_CHANNELS, # 3 (RGB)
            out_channels=FILTERS_1,     # 32
            kernel_size=KERNEL_SIZE,    # 3x3
            padding=1                   # Mantiene el tamaño espacial (32x32)
        )
        self.pool = nn.MaxPool2d(kernel_size=POOL_SIZE, stride=POOL_SIZE) # 2x2, reduce a la mitad

        # Bloque Convolucional 2: Conv2D -> MaxPooling2D
        self.conv2 = nn.Conv2d(
            in_channels=FILTERS_1,      # 32
            out_channels=FILTERS_2,     # 64
            kernel_size=KERNEL_SIZE,    # 3x3
            padding=1                   # Mantiene el tamaño espacial (16x16)
        )
        
        # ----------------------------------------
        # Clasificador (MLP)
        # ----------------------------------------
        
        # Calcular el tamaño lineal después de las convoluciones y pooling:
        # 32x32 -> Conv1(32 filtros) -> 32x32
        # 32x32 -> Pool1(2x2) -> 16x16
        # 16x16 -> Conv2(64 filtros) -> 16x16
        # 16x16 -> Pool2(2x2) -> 8x8
        # Tamaño de entrada a la capa densa: 64 filtros * 8 * 8 = 4096
        self.fc1 = nn.Linear(FILTERS_2 * 8 * 8, DENSE_UNITS_1) # 4096 -> 64
        
        # Capa de Salida
        self.fc2 = nn.Linear(DENSE_UNITS_1, NUM_CLASSES) # 64 -> 10
        
    def forward(self, x):
        # Bloque 1
        x = self.pool(F.relu(self.conv1(x))) # Conv -> ReLU -> Pool
        # Bloque 2
        x = self.pool(F.relu(self.conv2(x))) # Conv -> ReLU -> Pool
        
        # Flatten: Aplanar de (Batch_size, 64, 8, 8) a (Batch_size, 4096)
        x = torch.flatten(x, 1) 
        
        # Clasificador
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # Salida (no usamos softmax aquí, se aplica en la función de pérdida)
        return x

def build_cnn_model():
    """Crea e imprime la estructura del modelo."""
    model = SimpleCNN()
    print("Construyendo el modelo CNN (PyTorch)...")
    print(model)
    return model