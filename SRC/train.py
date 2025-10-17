# SRC/train.py (PyTorch Version)
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from SRC.config import EPOCHS, LEARNING_RATE
from tqdm import tqdm # Para una barra de progreso

def compile_and_train(model, train_loader, val_loader):
    """
    Define el optimizador, la función de pérdida y entrena el modelo.
    """
    # Usar GPU si está disponible
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Compilación: Optimizar y Función de Pérdida
    criterion = nn.CrossEntropyLoss() # Equivalente a categorical_crossentropy + softmax
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    print(f"\nIniciando entrenamiento en: {device}")
    
    for epoch in range(EPOCHS):
        # Fase de Entrenamiento
        model.train() # Pone el modelo en modo entrenamiento
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            
            # Forward + Backward + Optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train
        
        # Fase de Validación
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device, is_validation=True)
        
        print(f"\n[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        history['loss'].append(train_loss)
        history['accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)

    # Guardar el history
    if not os.path.exists("SALIDA"): os.makedirs("SALIDA")
    with open('SALIDA/training_history_pt.json', 'w') as f:
        json.dump(history, f)
    print("Historial de entrenamiento guardado en SALIDA/training_history_pt.json")

    return history

def save_model(model, path="SALIDA/cnn_cifar10_model_pt.pth"):
    """Guarda los pesos del modelo entrenado."""
    if not os.path.exists("SALIDA"): os.makedirs("SALIDA")
    torch.save(model.state_dict(), path)
    print(f"Modelo guardado exitosamente en {path}")

def evaluate_model(model, data_loader, criterion, device, is_validation=False):
    """Función auxiliar para evaluar tanto en validación como en prueba."""
    model.eval() # Pone el modelo en modo evaluación
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad(): # Desactiva el cálculo de gradientes
        for data in data_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(data_loader)
    accuracy = correct / total
    
    if not is_validation:
        print(f"\nResultados Finales en el Conjunto de Prueba:")
        print(f"Pérdida (Loss): {avg_loss:.4f}")
        print(f"Precisión (Accuracy): {accuracy:.4f}")
        
    return avg_loss, accuracy