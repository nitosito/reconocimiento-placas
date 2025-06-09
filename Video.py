import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import os
import time

# ====================================================================================
# --- CONFIGURACIÓN ---
# ====================================================================================
MODEL_PATH = 'runs/detect/train2/weights/best.pt'

# Para usar un archivo de video, pon la ruta: "videos/mi_video.mp4"
# Para usar la cámara web en vivo, pon el número: 0
VIDEO_SOURCE = 0 
# ====================================================================================

if __name__ == "__main__":
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"No se encuentra el archivo del modelo en: {MODEL_PATH}")

        print("Cargando modelos...")
        plate_detector = YOLO(MODEL_PATH)
        reader = easyocr.Reader(['es', 'en'], gpu=False)
        print("Modelos cargados.")

        video_capture = cv2.VideoCapture(VIDEO_SOURCE)
        if not video_capture.isOpened():
            raise ConnectionError(f"No se pudo abrir la fuente de video: {VIDEO_SOURCE}")

        prev_time = 0
        
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                print("Fin del video o error al leer el fotograma.")
                break

            # --- Detección y Reconocimiento por Fotograma ---
            results = plate_detector(frame)[0]

            for detection in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if score > 0.5:
                    plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                    
                    if plate_crop.size > 0:
                        ocr_result = reader.readtext(plate_crop)
                        if ocr_result:
                            plate_text = "".join([res[1] for res in ocr_result]).upper().replace(" ", "").replace(".", "").replace("-", "")
                            
                            # --- Visualización en Tiempo Real ---
                            color_fondo = (0, 100, 0) # Verde oscuro
                            color_texto = (255, 255, 255) # Blanco
                            
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color_fondo, 3)
                            
                            (text_width, text_height), baseline = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                            label_y1 = max(int(y1), text_height + 10) # Asegurarse de que no se salga por arriba
                            cv2.rectangle(frame, (int(x1), label_y1 - text_height - 10), (int(x1) + text_width, label_y1 - 5), color_fondo, cv2.FILLED)
                            cv2.putText(frame, plate_text, (int(x1), label_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_texto, 2)
            
            # --- Cálculo y Visualización de FPS ---
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Muestra el fotograma procesado
            cv2.imshow("Deteccion en Tiempo Real - Presiona 'q' para salir", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"\nHa ocurrido un error: {e}")