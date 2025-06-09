# super_app.py (Versión Definitiva)

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Importamos las funciones de nuestro motor
# Esta línea ahora encontrará las funciones correctas en el archivo que acabamos de guardar
from anpr_engine import process_image_for_dashboard, process_frame_for_realtime

# --- Configuración de la Página de Streamlit ---
st.set_page_config(layout="wide", page_title="ANPR Super App", page_icon="📸")

# --- Título Principal ---
st.title("RED CONVOLUCIONAL PARA EL RECONOCIMIENTO DE PLACAS VEHICULARES COLOMBIANAS")

# --- Creación de las Pestañas ---
tab1, tab2 = st.tabs(["📁 Análisis Detallado de Imagen", "📹 Detección en Tiempo Real (Webcam)"])

# --- Contenido de la Pestaña 1: Análisis de Imagen ---
with tab1:
    st.header("Sube una imagen para un análisis completo tipo dashboard")
    uploaded_file = st.file_uploader("Elige una imagen de un vehículo...", type=["jpg", "png", "jpeg"], key="uploader")

    if uploaded_file is not None:
        # Convertimos el archivo subido a una imagen que OpenCV pueda leer
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_to_process = cv2.imdecode(file_bytes, 1)

        if st.button("Generar Dashboard de Análisis", use_container_width=True):
            with st.spinner('Realizando análisis profundo...'):
                # Llamamos a nuestra función que crea el dashboard completo
                dashboard_image, detected_text = process_image_for_dashboard(image_to_process)
            
            st.success("¡Análisis completado!")
            st.image(dashboard_image, channels="BGR", caption="Dashboard de Resultados")
            st.subheader(f"Texto Detectado: `{detected_text}`")

# --- Contenido de la Pestaña 2: Detección en Tiempo Real ---
with tab2:
    st.header("Detección en vivo desde tu cámara web")
    st.write("Presiona 'Start' para activar tu cámara. La detección se realizará en tiempo real.")
    st.warning("Asegúrate de permitir el acceso a tu cámara en el navegador.")

    # Clase para procesar cada fotograma que llega desde la cámara
    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            processed_img = process_frame_for_realtime(img)
            return processed_img

    # El componente que activa la cámara y muestra el video
    webrtc_streamer(
        key="realtime_detection",
        video_transformer_factory=VideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
