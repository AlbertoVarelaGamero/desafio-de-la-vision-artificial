# main.py
import sys
import os
import torch

# Añadir el directorio actual al path para que Python encuentre 'SRC'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from SRC.data_loader import load_and_preprocess_data, visualize_cifar10_samples
from SRC.model import build_cnn_model
from SRC.train import compile_and_train, save_model
from SRC.evaluate import final_evaluate
from SRC.report import plot_training_history, display_analysis_question_answer

def main():
    print("--- INICIO DE ACTIVIDAD 3: CORTEX VISUAL ARTIFICIAL (CNN) CON PYTORCH ---")
    
    # 0. Configuración inicial
    if not os.path.exists("SALIDA"):
        os.makedirs("SALIDA")
        
    # 1. Fase 1: Preparación del Dataset Visual (CIFAR-10)
    # trainset se usa para la visualización de la muestra
    train_loader, val_loader, test_loader, trainset = load_and_preprocess_data()
    visualize_cifar10_samples(trainset)

    # 2. Fase 2: Arquitectura del "Córtex Visual" (Construcción del Modelo)
    model = build_cnn_model()

    # 3. Fase 3: Entrenamiento y Evaluación
    compile_and_train(model, train_loader, val_loader)
    
    # Guardar el modelo
    save_model(model)

    # Evaluación Final
    final_evaluate(model, test_loader)
    
    # Generar gráficas de reporte
    plot_training_history()
    
    # Responder a la pregunta de análisis (para el informe)
    display_analysis_question_answer()

    print("--- FIN DE LA ACTIVIDAD ---")

if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        if 'torch' in str(e) or 'torchvision' in str(e):
            print("\n[ERROR CRÍTICO] PyTorch o Torchvision no está instalado. Ejecuta 'pip install torch torchvision numpy matplotlib tqdm'")
        else:
            print(f"\n[ERROR DE IMPORTACIÓN] Por favor, revisa la consola. Error: {e}")
            sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR GENERAL] Ocurrió un error no manejado durante la ejecución: {e}")
        sys.exit(1)