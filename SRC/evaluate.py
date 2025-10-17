# SRC/evaluate.py (PyTorch Version)
import torch
import torch.nn as nn
from SRC.train import evaluate_model

def final_evaluate(model, test_loader):
    """
    Carga la función de evaluación y ejecuta la evaluación final.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    
    model.to(device)
    loss, accuracy = evaluate_model(model, test_loader, criterion, device)
    
    return loss, accuracy