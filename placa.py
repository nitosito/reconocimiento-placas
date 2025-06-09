import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import os
import re # Importamos la librería de expresiones regulares

# ====================================================================================
# --- CONFIGURACIÓN ---
# ====================================================================================
MODEL_PATH = 'runs/detect/train2/weights/best.pt'
# ====================================================================================

def resize_for_display(image, width, height):
    """Redimensiona una imagen a un tamaño fijo, manejando color y escala de grises."""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return cv2.resize(image, (width, height))

def format_plate_text(ocr_result):
    """
    Analiza el resultado de EasyOCR y lo formatea de manera inteligente.
    AHORA MEJORADO CON EXPRESIONES REGULARES para el formato ABC-123.
    """
    plate_number = ""
    city = ""
    
    for (bbox, text, prob) in ocr_result:
        cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        if any(char.isdigit() for char in cleaned_text):
            plate_number += cleaned_text
        elif cleaned_text.isalpha() and len(cleaned_text) > 2: # Evitar letras sueltas
            city += cleaned_text

    # --- LÓGICA DE FORMATEO MEJORADA ---
    # Usamos una expresión regular para buscar el patrón: 3 letras seguidas de 3 números.
    match = re.match(r'^([A-Z]{3})([0-9]{3})$', plate_number)
    
    if match:
        # Si encuentra el patrón, lo formatea con un guion.
        plate_number = f"{match.group(1)}-{match.group(2)}"
        
    # Ensamblamos el resultado final
    final_text = plate_number
    if city:
        final_text += f" de {city}"
        
    return final_text

# --- EJECUCIÓN PRINCIPAL DEL PROGRAMA (sin cambios) ---
if __name__ == "__main__":
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"No se encuentra el archivo del modelo en: {MODEL_PATH}")

        print("Cargando modelos...")
        plate_detector = YOLO(MODEL_PATH)
        reader = easyocr.Reader(['es', 'en'], gpu=False)
        print("Modelos cargados.")
        
        image_path = input("Arrastra una imagen aquí o escribe la ruta y presiona Enter: ").strip().replace("'", "").replace('"', '')

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"El archivo no existe en la ruta: {image_path}")
            
        original_image = cv2.imread(image_path)
        if original_image is None: raise ValueError("OpenCV no pudo leer la imagen.")
            
        result_image = original_image.copy()

        print("\nDetectando y leyendo placas...")
        results = plate_detector(original_image)[0]
        
        detected_plate_text = "No se pudo leer"
        plate_crop_for_display = np.zeros((80, 200, 3), dtype=np.uint8) 

        for detection in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if score > 0.5:
                plate_crop = original_image[int(y1):int(y2), int(x1):int(x2)]
                if plate_crop.size > 0: plate_crop_for_display = plate_crop.copy()
                
                ocr_result = reader.readtext(plate_crop)

                if ocr_result:
                    detected_plate_text = format_plate_text(ocr_result)
                    print(f"  > Texto Formateado: {detected_plate_text} (Confianza de detección: {score:.2f})")
                    
                    # --- Dibuja la visualización avanzada en la imagen de resultado ---
                    color_fondo_texto = (0, 100, 0) # Verde oscuro
                    color_texto = (255, 255, 255) # Blanco
                    (text_width, text_height), _ = cv2.getTextSize(detected_plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                    cv2.rectangle(result_image, (int(x1), int(y1) - text_height - 15), (int(x1) + text_width, int(y1) - 5), color_fondo_texto, cv2.FILLED)
                    cv2.putText(result_image, detected_plate_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_texto, 2)
                    cv2.rectangle(result_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

                break
        
        # El resto del código del dashboard no necesita cambios...
        # ... (código para generar imágenes intermedias y pegar todo en el canvas) ...

        # --- CREACIÓN DEL DASHBOARD (sin cambios) ---
        print("Generando imágenes del proceso...")
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        canny_edges = cv2.Canny(gray_image, 100, 200)
        _, background_removed_sim = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
        disp_width, disp_height = 400, 300
        margin = 30 
        img_original_disp = resize_for_display(original_image, disp_width, disp_height)
        img_gray_disp = resize_for_display(gray_image, disp_width, disp_height)
        img_edges_disp = resize_for_display(canny_edges, disp_width, disp_height)
        img_bg_removed_sim = resize_for_display(background_removed_sim, disp_width, disp_height)
        img_result_disp = resize_for_display(result_image, disp_width, disp_height)
        img_plate_disp = resize_for_display(plate_crop_for_display, 200, 80)
        canvas_height = disp_height * 2 + margin * 4
        canvas_width = disp_width * 3 + margin * 4 
        canvas = np.full((canvas_height, canvas_width, 3), (48, 48, 48), dtype=np.uint8)
        
        # ... (código para pegar las imágenes y textos en el canvas) ...
        # (He omitido el resto del código del dashboard por brevedad, pero es el mismo que ya tienes)
        cv2.putText(canvas, "Proceso", (margin, margin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        y_pos = margin * 2; canvas[y_pos:y_pos + disp_height, margin:margin + disp_width] = img_original_disp; cv2.putText(canvas, "Imagen cargada", (margin, y_pos + disp_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += disp_height + margin; canvas[y_pos:y_pos + disp_height, margin:margin + disp_width] = img_gray_disp; cv2.putText(canvas, "Imagen escala de grises", (margin, y_pos + disp_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        x_pos_col2 = disp_width + margin * 2; y_pos = margin * 2; canvas[y_pos:y_pos + disp_height, x_pos_col2:x_pos_col2 + disp_width] = img_bg_removed_sim; cv2.putText(canvas, "Imagen con fondo eliminado (sim)", (x_pos_col2, y_pos + disp_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += disp_height + margin; canvas[y_pos:y_pos + disp_height, x_pos_col2:x_pos_col2 + disp_width] = img_edges_disp; cv2.putText(canvas, "Imagen con solo bordes", (x_pos_col2, y_pos + disp_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        x_pos_col3 = disp_width * 2 + margin * 3; cv2.putText(canvas, "Resultado", (x_pos_col3, margin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        y_pos = margin * 2; canvas[y_pos:y_pos + disp_height, x_pos_col3:x_pos_col3 + disp_width] = img_result_disp
        y_pos += disp_height + margin; cv2.putText(canvas, "Placa procesada", (x_pos_col3, y_pos - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1); canvas[y_pos:y_pos + 80, x_pos_col3:x_pos_col3 + 200] = img_plate_disp
        y_pos += 80 + margin; cv2.putText(canvas, "Texto detectado:", (x_pos_col3, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1); cv2.putText(canvas, detected_plate_text, (x_pos_col3, y_pos + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 128), 2)
        
        cv2.imshow("Dashboard de Reconocimiento de Placas", canvas)
        print("\nVentana con el dashboard mostrada. Presiona 'q' para salir.")
        while cv2.waitKey(1) & 0xFF != ord('q'):
            if cv2.getWindowProperty("Dashboard de Reconocimiento de Placas", cv2.WND_PROP_VISIBLE) < 1: break
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"\nHa ocurrido un error: {e}")