# SRC/report.py (PyTorch Version)
import matplotlib.pyplot as plt
import json
import os

def plot_training_history(history_path="SALIDA/training_history_pt.json"):
    """
    Grafica la precisión y la pérdida de entrenamiento y validación.
    """
    if not os.path.exists(history_path):
        print(f"Error: No se encontró el historial de entrenamiento en {history_path}")
        return

    with open(history_path, 'r') as f:
        history = json.load(f)

    # PyTorch history keys: 'accuracy', 'loss', 'val_accuracy', 'val_loss'
    epochs = range(1, len(history['accuracy']) + 1)
    
    # 1. Gráfico de Precisión
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['accuracy'], 'b', label='Precisión Entrenamiento')
    plt.plot(epochs, history['val_accuracy'], 'r', label='Precisión Validación')
    plt.title('Precisión de Entrenamiento y Validación')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()

    # 2. Gráfico de Pérdida
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['loss'], 'b', label='Pérdida Entrenamiento')
    plt.plot(epochs, history['val_loss'], 'r', label='Pérdida Validación')
    plt.title('Pérdida de Entrenamiento y Validación')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()

    plt.tight_layout()
    plot_path = "SALIDA/training_metrics_pt.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"\nGráficas de rendimiento guardadas en {plot_path}")

def display_analysis_question_answer():
    """Responde a la Pregunta de Análisis 1."""
    print("\n" + "="*80)
    print("Pregunta de Análisis 1: Respuesta")
    print("="*80)
    
    print("El diagrama de procesamiento de visión artificial tradicional (Input, Preprocessing, Feature extraction, Classifier) se mapea a la CNN de la siguiente manera:\n")
    
    print("1. Input (Entrada):")
    print("\t- Corresponde a la entrada del tensor a la capa `conv1` (`in_channels=3`).")

    print("2. Preprocessing (Preprocesamiento):")
    print("\t- Corresponde a la transformación **`transforms.Normalize`** aplicada en `data_loader.py` antes de cargar los datos.")

    print("3. Feature Extraction (Extracción de Características):")
    print("\t- Corresponde a las capas **`nn.Conv2d`** y **`nn.MaxPool2d`** apiladas.")
    print("\t- **`nn.Conv2d`** extrae automáticamente características (bordes, texturas, formas) a través de filtros entrenables.")
    print("\t- **`nn.MaxPool2d`** resume y reduce la dimensionalidad de estas características.")

    print("4. Classifier (Clasificador):")
    print("\t- Corresponde al aplanamiento (`torch.flatten`) seguido de las capas **`nn.Linear`** (`fc1` y `fc2`).")
    
    print("\nAutomatización de 'Feature Extraction':")
    print("La CNN automatiza la 'Feature Extraction' porque, a diferencia del enfoque tradicional donde un humano o un algoritmo manual (ej: SIFT, HOG) debe diseñar las características, **la CNN aprende los filtros (kernels) óptimos de forma automática durante el entrenamiento** (propagación hacia atrás y descenso de gradiente). Las capas Conv2D de bajo nivel aprenden a detectar bordes, y las capas superiores aprenden a combinar esos bordes en formas y patrones complejos, eliminando la necesidad de un preprocesamiento manual y específico para características.")
    print("="*80)