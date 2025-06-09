# anpr_engine.py (Versión Definitiva)

import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import re
import os

# --- Carga de Modelos (Se hace una sola vez al iniciar) ---
print("Cargando modelos ANPR (esto puede tardar un momento)...")
try:
    # Asegúrate de que esta ruta a tu modelo sea correcta
    MODEL_PATH = 'runs/detect/train2/weights/best.pt'
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"El archivo del modelo no se encuentra en la ruta especificada: {MODEL_PATH}")
    
    plate_detector = YOLO(MODEL_PATH)
    reader = easyocr.Reader(['es', 'en'], gpu=False)
    print("Modelos cargados exitosamente.")
except Exception as e:
    print(f"Error crítico al cargar los modelos: {e}")
    plate_detector = None
    reader = None

def resize_for_display(image, width, height):
    """Redimensiona una imagen a un tamaño fijo, manejando color y escala de grises."""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return cv2.resize(image, (width, height))

def format_plate_text(ocr_result):
    """Analiza el resultado de EasyOCR y lo formatea de manera inteligente."""
    plate_number = ""
    city = ""
    for _, text, _ in ocr_result:
        cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
        if any(char.isdigit() for char in cleaned_text):
            plate_number += cleaned_text
        elif cleaned_text.isalpha() and len(cleaned_text) > 2:
            city += cleaned_text
    
    match = re.match(r'^([A-Z]{3})([0-9]{3})$', plate_number)
    if match:
        plate_number = f"{match.group(1)}-{match.group(2)}"
        
    final_text = plate_number
    if city:
        final_text += f" de {city}"
    return final_text if final_text else "No se pudo leer"

# --- FUNCIÓN 1: Para el Dashboard Detallado ---
def process_image_for_dashboard(image_object):
    """
    Toma un objeto de imagen de OpenCV, la procesa y devuelve
    el canvas completo del dashboard como una imagen y el texto detectado.
    """
    if not plate_detector or not reader:
        raise RuntimeError("Los modelos no se cargaron correctamente.")
        
    original_image = image_object
    result_image = original_image.copy()
    detected_plate_text = "No se detecto placa"
    plate_crop_for_display = np.zeros((80, 200, 3), dtype=np.uint8) 

    results = plate_detector(original_image)[0]
    for detection in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if score > 0.5:
            plate_crop = original_image[int(y1):int(y2), int(x1):int(x2)]
            if plate_crop.size > 0:
                plate_crop_for_display = plate_crop.copy()
            ocr_result = reader.readtext(plate_crop)
            if ocr_result:
                detected_plate_text = format_plate_text(ocr_result)
                cv2.rectangle(result_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
            break

    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    canny_edges = cv2.Canny(gray_image, 100, 200)
    _, background_removed_sim = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    
    disp_width, disp_height = 400, 300
    margin = 30 
    img_original_disp = resize_for_display(original_image, disp_width, disp_height)
    img_gray_disp = resize_for_display(gray_image, disp_width, disp_height)
    img_edges_disp = resize_for_display(canny_edges, disp_width, disp_height)
    img_bg_removed_disp = resize_for_display(background_removed_sim, disp_width, disp_height)
    img_result_disp = resize_for_display(result_image, disp_width, disp_height)
    img_plate_disp = resize_for_display(plate_crop_for_display, 200, 80)

    canvas_height = disp_height * 2 + margin * 4
    canvas_width = disp_width * 3 + margin * 4 
    canvas = np.full((canvas_height, canvas_width, 3), (48, 48, 48), dtype=np.uint8)
    
    cv2.putText(canvas, "Proceso", (margin, margin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    y_pos = margin * 2; canvas[y_pos:y_pos + disp_height, margin:margin + disp_width] = img_original_disp; cv2.putText(canvas, "Imagen cargada", (margin, y_pos + disp_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_pos += disp_height + margin; canvas[y_pos:y_pos + disp_height, margin:margin + disp_width] = img_gray_disp; cv2.putText(canvas, "Imagen escala de grises", (margin, y_pos + disp_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    x_pos_col2 = disp_width + margin * 2; y_pos = margin * 2; canvas[y_pos:y_pos + disp_height, x_pos_col2:x_pos_col2 + disp_width] = img_bg_removed_disp; cv2.putText(canvas, "Imagen con fondo eliminado (sim)", (x_pos_col2, y_pos + disp_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_pos += disp_height + margin; canvas[y_pos:y_pos + disp_height, x_pos_col2:x_pos_col2 + disp_width] = img_edges_disp; cv2.putText(canvas, "Imagen con solo bordes", (x_pos_col2, y_pos + disp_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    x_pos_col3 = disp_width * 2 + margin * 3; cv2.putText(canvas, "Resultado", (x_pos_col3, margin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    y_pos = margin * 2; canvas[y_pos:y_pos + disp_height, x_pos_col3:x_pos_col3 + disp_width] = img_result_disp
    y_pos += disp_height + margin; cv2.putText(canvas, "Placa procesada", (x_pos_col3, y_pos - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1); canvas[y_pos:y_pos + 80, x_pos_col3:x_pos_col3 + 200] = img_plate_disp
    y_pos += 80 + margin; cv2.putText(canvas, "Texto detectado:", (x_pos_col3, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1); cv2.putText(canvas, detected_plate_text, (x_pos_col3, y_pos + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 128), 2)
    
    return canvas, detected_plate_text

# --- FUNCIÓN 2: Para el Video en Tiempo Real ---
def process_frame_for_realtime(frame):
    """
    Toma un fotograma de video y devuelve el mismo fotograma
    con los resultados dibujados encima para una visualización rápida.
    """
    results = plate_detector(frame, verbose=False)[0]
    for detection in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if score > 0.5:
            plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
            if plate_crop.size > 0:
                ocr_result = reader.readtext(plate_crop)
                if ocr_result:
                    plate_text = format_plate_text(ocr_result)
                    (text_width, text_height), _ = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                    cv2.rectangle(frame, (int(x1), int(y1) - text_height - 15), (int(x1) + text_width, int(y1) - 5), (0, 100, 0), cv2.FILLED)
                    cv2.putText(frame, plate_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
    return frame