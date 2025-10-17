# app_streamlit.py

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import sys
import os

# --- Configuración de Rutas y Modelo ---

# Asegurar que Streamlit encuentre la carpeta SRC
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar la arquitectura del modelo (asumiendo que tiene Dropout implementado)
from SRC.model import SimpleCNN 
from SRC.config import CIFAR10_LABELS

MODEL_PATH = "SALIDA/cnn_cifar10_model_pt.pth"

# Definir las transformaciones necesarias para la inferencia
inference_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# --- Función de Carga de Modelo y Caché ---

@st.cache_resource
def load_model():
    """Carga el modelo PyTorch entrenado usando caché de Streamlit."""
    try:
        # Asegúrate de que SimpleCNN usa la arquitectura con Dropout
        model = SimpleCNN()
        
        if os.path.exists(MODEL_PATH):
            # Cargar los pesos en la CPU
            # Es importante usar map_location='cpu' si entrenaste sin GPU.
            model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
            model.eval() # Poner el modelo en modo evaluación
            return model
        else:
            # Mostrar error si el modelo no existe (si main.py no se ha ejecutado)
            st.error(f"ERROR: No se encontró el modelo entrenado en {MODEL_PATH}.")
            st.warning("Por favor, ejecuta 'python main.py' primero para entrenar el modelo.")
            return None
    except Exception as e:
        st.error(f"Error al cargar o inicializar el modelo: {e}")
        return None

# --- Función de Inferencia ---

def classify_image_st(uploaded_file):
    """Procesa el archivo de Streamlit, clasifica y devuelve los resultados."""
    if uploaded_file is not None:
        # Abrir la imagen subida por el usuario
        img = Image.open(uploaded_file).convert('RGB')
        
        # 1. Preprocesar y preparar tensor
        tensor = inference_transform(img).unsqueeze(0) 

        # 2. Inferir
        model = load_model()
        if model is None:
            return None, None
            
        with torch.no_grad():
            output = model(tensor)

        # 3. Obtener probabilidades (Softmax)
        probabilities = F.softmax(output, dim=1)[0]
        
        # 4. Obtener las 5 predicciones principales
        top_p, top_class = probabilities.topk(5, dim=0)
        
        # 5. Formatear resultados para Streamlit (Lista de tuplas para la tabla)
        results = [(CIFAR10_LABELS[top_class[i]], top_p[i].item()) for i in range(top_p.size(0))]
        
        return img, results
    return None, None

# --- Interfaz Principal de Streamlit ---

# Configuración de la página
st.set_page_config(
    page_title="Córtex Visual Artificial (CNN)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título y Descripción
st.title("🧠 Córtex Visual Artificial (CNN) para CIFAR-10")
st.markdown("Clasificador de imágenes entrenado en el dataset CIFAR-10 (10 clases).")
st.markdown("---")

# Carga de archivos y panel de resultados en una estructura de columnas
col1, col2 = st.columns([1, 2])

with col1:
    st.header("1. Cargar Imagen")
    uploaded_file = st.file_uploader(
        "Sube una imagen (PNG, JPG, JPEG)", 
        type=["png", "jpg", "jpeg"]
    )
    
    # Mostrar la imagen subida
    if uploaded_file is not None:
        st.subheader("Imagen de Entrada")
        
        # Llamada a la función de clasificación
        image_to_display, predictions = classify_image_st(uploaded_file)
        
        if image_to_display is not None:
            st.image(image_to_display, caption="Imagen Original", use_column_width=True)

with col2:
    st.header("2. Resultados del Modelo")
    
    if uploaded_file is None:
        st.info("Sube una imagen en el panel de la izquierda para iniciar la clasificación.")
    elif predictions:
        st.subheader("Análisis de Predicción")
        
        # Mostrar la predicción principal
        top_label = predictions[0][0]
        top_score = predictions[0][1] * 100
        
        # Usamos éxito (success) si la confianza es superior al 50%
        if top_score > 50:
            st.success(f"**Clase Predicha:** {top_label}")
        else:
            st.warning(f"**Clase Predicha:** {top_label} (Confianza baja)")
            
        st.metric(label="Confianza Principal", value=f"{top_score:.2f} %")

        st.subheader("Top 5 Probabilidades")
        
        # Mostrar las 5 predicciones en una tabla
        data = {
            'Clase': [p[0] for p in predictions],
            'Probabilidad (%)': [f"{p[1]*100:.2f}" for p in predictions]
        }
        st.dataframe(data, use_container_width=True, hide_index=True)

# Información adicional en la barra lateral
with st.sidebar:
    st.header("Detalles del Modelo")
    st.markdown("""
    - **Arquitectura:** CNN Simple (mejorada con Dropout).
    - **Técnicas Anti-Sobreajuste:** **Dropout** y **Aumentación de Datos**.
    - **Dataset:** CIFAR-10 (10 clases).
    - **Framework:** PyTorch
    - **Interfaz:** Streamlit
    """)
    st.warning("La resolución original de las imágenes es 32x32. Las fotos reales se reescalan, lo que puede limitar la precisión.")