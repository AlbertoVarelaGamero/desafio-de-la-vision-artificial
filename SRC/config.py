# SRC/config.py (PyTorch Version)

# Constantes del Dataset
NUM_CLASSES = 10
INPUT_CHANNELS = 3 # RGB
IMAGE_SIZE = 32

# Constantes del Modelo
KERNEL_SIZE = 3
POOL_SIZE = 2
FILTERS_1 = 32
FILTERS_2 = 64
DENSE_UNITS_1 = 64

# Constantes de Entrenamiento
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

# Etiquetas de las Clases de CIFAR-10 (para visualización)
CIFAR10_LABELS = [
    "airplane", "automobile", "bird", "cat", "deer", 
    "dog", "frog", "horse", "ship", "truck"
]