import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# --- Configuración de MediaPipe ---
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True, min_detection_confidence=0.6)

# Buffer para suavizar el resultado
buffer_emociones = deque(maxlen=15)

# --- Configuración de Interfaz ---
ANCHO_CAM, ALTO_CAM = 640, 480
ANCHO_PANEL = 300
canvas_w = ANCHO_CAM + ANCHO_PANEL

def obtener_emocion(lista):
    if not lista: return "NEUTRO"
    return max(set(lista), key=lista.count)

def get_punto(face, idx):
    """Obtiene las coordenadas de un landmark en píxeles."""
    return np.array([face.landmark[idx].x * ANCHO_CAM, face.landmark[idx].y * ALTO_CAM])

cam = cv2.VideoCapture(0)

try:
    if not cam.isOpened():
        raise RuntimeError("No se pudo acceder a la cámara")

    while True:
        ret, frame = cam.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (ANCHO_CAM, ALTO_CAM))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        # Crear interfaz
        ui = np.zeros((ALTO_CAM, canvas_w, 3), dtype=np.uint8)
        ui[:, :ANCHO_CAM] = frame
        ui[:, ANCHO_CAM:] = (40, 40, 40) # Fondo panel

        emocion_detectada = "NEUTRO"

        if res.multi_face_landmarks:
            for face in res.multi_face_landmarks:
                # Puntos de referencia
                ojo_izq, ojo_der = get_punto(face, 133), get_punto(face, 362)
                dist_ojos = np.linalg.norm(ojo_izq - ojo_der) # Nuestra "regla" universal

                # Evitar división por cero si los landmarks están mal detectados
                if dist_ojos < 1.0:
                    continue

                # Dibujar puntos de depuración (para que veas que sí detecta)
                for idx in [13, 14, 61, 291, 52, 282, 0]:
                    p = get_punto(face,idx)
                    cv2.circle(ui, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1)

                # --- MÉTRICAS RELATIVAS A LA DISTANCIA DE OJOS ---
                # Boca
                apertura_boca = np.linalg.norm(get_punto(face,13) - get_punto(face,14)) / dist_ojos
                ancho_boca = np.linalg.norm(get_punto(face,61) - get_punto(face,291)) / dist_ojos

                # Cejas (Enojo/Tristeza)
                # Distancia desde el centro de la ceja al ojo
                ceja_izq = np.linalg.norm(get_punto(face,52) - get_punto(face,159)) / dist_ojos
                ceja_der = np.linalg.norm(get_punto(face,282) - get_punto(face,386)) / dist_ojos
                altura_cejas = (ceja_izq + ceja_der) / 2

                # Labios (Comisuras para sonrisa o tristeza)
                comisura_izq = get_punto(face,61)
                comisura_der = get_punto(face,291)
                centro_boca = get_punto(face,0)
                # Si las comisuras están más arriba que el centro (Y es menor), es sonrisa
                curvatura = (centro_boca[1] - (comisura_izq[1] + comisura_der[1]) / 2) / dist_ojos

                # --- REGLAS DE DETECCIÓN (Umbrales más generosos) ---
                if apertura_boca > 0.4:
                    emocion_detectada = "SORPRESA"
                elif curvatura > 0.05 or ancho_boca > 0.9:
                    emocion_detectada = "FELIZ"
                elif altura_cejas < 0.18: # Cejas bajas = enojo
                    emocion_detectada = "ENOJADO"
                elif curvatura < -0.05: # Comisuras caídas
                    emocion_detectada = "TRISTE"
                else:
                    emocion_detectada = "NEUTRO"

                break  # Solo procesar la primera cara detectada

        buffer_emociones.append(emocion_detectada)
        final_emo = obtener_emocion(list(buffer_emociones))

        # --- PANEL DERECHO ---
        textos = {
            "FELIZ": ["SONRISA DETECTADA", "Te ves genial.", "Sigue con esa actitud!"],
            "TRISTE": ["ESTADO: TRISTE", "Pareces decaido.", "Quieres hablar con alguien?"],
            "ENOJADO": ["ESTADO: ENOJADO", "Relaja el rostro.", "Respira y cuenta hasta 10."],
            "SORPRESA": ["¡OH! SORPRESA", "Viste algo increible?", "Respira con calma."],
            "NEUTRO": ["MODO: NEUTRO", "Rostro relajado.", "Todo fluye con calma."]
        }

        info = textos[final_emo]
        cv2.putText(ui, info[0], (ANCHO_CAM + 20, 60), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(ui, info[1], (ANCHO_CAM + 20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(ui, info[2], (ANCHO_CAM + 20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Asistente Emocional", ui)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

except KeyboardInterrupt:
    print("\nPrograma interrumpido por el usuario")
except RuntimeError as e:
    print(f"Error de ejecución: {e}")
except Exception as e:
    print(f"Error inesperado: {e}")
finally:
    cam.release()
    cv2.destroyAllWindows()