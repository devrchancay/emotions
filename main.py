import os
import cv2
import mediapipe as mp
import numpy as np
from collections import deque, Counter
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Configuración de MediaPipe ---
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True, min_detection_confidence=0.6)

# Buffer para suavizar el resultado
buffer_emociones = deque(maxlen=15)

# --- Configuración de Interfaz ---
ANCHO_CAM, ALTO_CAM = 640, 480
ANCHO_PANEL = 300
canvas_w = ANCHO_CAM + ANCHO_PANEL
FPS_OBJETIVO = 30
FRAME_DELAY = int(1000 / FPS_OBJETIVO)  # ms entre frames
DEBUG = os.getenv("DEBUG", "0").lower() in ("1", "true", "yes")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Log de emociones en memoria ---
log_emociones = []  # Lista de (timestamp, emocion)
ultima_emocion = None  # Última emoción registrada en el log
msg_guardado = ""   # Mensaje temporal de ruta guardada
msg_guardado_timer = 0  # Frames restantes para mostrar mensaje

# --- Umbrales de detección de emociones ---
UMBRAL_SORPRESA_BOCA = 0.55      # Apertura de boca para sorpresa (más alto = menos sensible)
UMBRAL_FELIZ_CURVATURA = 0.05    # Curvatura de labios para felicidad
UMBRAL_ENOJADO_CEJAS = 0.45      # Distancia entre cejas (menor = ceño fruncido)
UMBRAL_TRISTE_CURVATURA = -0.55  # Curvatura de labios para tristeza (más negativo = menos sensible)

def obtener_emocion(lista):
    if not lista: return "NEUTRO"
    # Counter cuenta todas las ocurrencias en una sola pasada, mientras que el método anterior llamaba lista.count() para cada elemento único.
    return Counter(lista).most_common(1)[0][0]

def get_punto(face, idx):
    """Obtiene las coordenadas de un landmark en píxeles."""
    return np.array([face.landmark[idx].x * ANCHO_CAM, face.landmark[idx].y * ALTO_CAM])

# --- Malla geométrica facial (wireframe) ---
MALLA_VERTICES = [
    10, 21, 251,                          # Frente, sienes
    46, 276, 107, 336,                    # Cejas externas e internas
    33, 133, 263, 362,                    # Esquinas de ojos
    159, 386,                             # Párpados superiores
    145, 374,                             # Párpados inferiores
    6, 1, 98, 327,                        # Nariz
    116, 345,                             # Pómulos
    61, 291, 0, 17,                       # Boca
    132, 361, 58, 288, 152,              # Mandíbula y mentón
]

MALLA_CONEXIONES = [
    # Frente → cejas
    (10, 107), (10, 336), (10, 21), (10, 251),
    (21, 46), (251, 276),
    (46, 107), (276, 336), (107, 336),
    # Cejas → ojos
    (46, 33), (107, 133), (276, 263), (336, 362),
    (33, 159), (159, 133), (263, 386), (386, 362),
    (33, 145), (145, 133), (263, 374), (374, 362),
    # Ojos → nariz
    (133, 6), (362, 6), (6, 1), (1, 98), (1, 327),
    (133, 98), (362, 327),
    # Pómulos
    (33, 116), (263, 345), (116, 98), (345, 327),
    # Boca
    (98, 61), (327, 291), (116, 61), (345, 291),
    (61, 0), (291, 0), (61, 17), (291, 17), (0, 1),
    # Mandíbula
    (21, 132), (251, 361), (132, 116), (361, 345),
    (132, 58), (361, 288), (58, 61), (288, 291),
    (58, 152), (288, 152), (17, 152),
]

COLOR_LINEA = (0, 255, 0)       # Verde brillante
COLOR_GLOW = (0, 100, 0)        # Verde oscuro para resplandor
COLOR_PUNTO = (180, 255, 180)   # Verde claro para vértices


def dibujar_rostro(ui, face):
    """Dibuja malla geométrica wireframe sobre el rostro con glow suave en vértices."""
    # Capa separada para el glow de los vértices
    glow = np.zeros_like(ui)
    for idx in MALLA_VERTICES:
        p = tuple(get_punto(face, idx).astype(int))
        cv2.circle(glow, p, 8, (0, 180, 0), -1, cv2.LINE_AA)
    glow = cv2.GaussianBlur(glow, (15, 15), 0)
    cv2.add(ui, glow, ui)

    # Dibujar líneas de conexión
    for a, b in MALLA_CONEXIONES:
        pa = tuple(get_punto(face, a).astype(int))
        pb = tuple(get_punto(face, b).astype(int))
        cv2.line(ui, pa, pb, COLOR_LINEA, 1, cv2.LINE_AA)

    # Dibujar centro del vértice (punto pequeño nítido)
    for idx in MALLA_VERTICES:
        p = tuple(get_punto(face, idx).astype(int))
        cv2.circle(ui, p, 2, COLOR_PUNTO, -1, cv2.LINE_AA)


def detectar_camaras(max_index=10):
    """Detecta las cámaras disponibles probando índices."""
    camaras = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                nombre = f"Camara {i}"
                backend = cap.getBackendName()
                if backend:
                    nombre += f" ({backend})"
                camaras.append((i, nombre))
            cap.release()
    return camaras


def pantalla_inicio(camaras):
    """Muestra pantalla de bienvenida y captura nombre, apellido y cámara."""
    nombre = ""
    apellido = ""
    campo_activo = 0  # 0 = nombre, 1 = apellido, 2 = camara
    cam_seleccionada = 0
    parpadeo = 0

    while True:
        ui = np.zeros((ALTO_CAM, canvas_w, 3), dtype=np.uint8)
        ui[:] = (40, 40, 40)

        # Titulo
        cv2.putText(ui, "ANALISIS DE EMOCIONES", (canvas_w // 2 - 230, 60),
                     cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 2)

        # Linea decorativa
        cv2.line(ui, (canvas_w // 2 - 230, 75), (canvas_w // 2 + 230, 75), (0, 255, 255), 1)

        # Descripcion
        desc = [
            "Sistema de deteccion de emociones en tiempo real",
            "mediante analisis facial con inteligencia artificial.",
            "Detecta: Felicidad, Tristeza, Enojo, Sorpresa y Neutro."
        ]
        for i, linea in enumerate(desc):
            cv2.putText(ui, linea, (canvas_w // 2 - 250, 110 + i * 25),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        # Campo nombre
        color_nombre = (0, 255, 255) if campo_activo == 0 else (120, 120, 120)
        cv2.putText(ui, "Nombre:", (canvas_w // 2 - 180, 210),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_nombre, 1)
        cv2.rectangle(ui, (canvas_w // 2 - 180, 218), (canvas_w // 2 + 180, 248), color_nombre, 1)
        cursor_n = "|" if campo_activo == 0 and parpadeo % 40 < 20 else ""
        cv2.putText(ui, nombre + cursor_n, (canvas_w // 2 - 170, 242),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # Campo apellido
        color_apellido = (0, 255, 255) if campo_activo == 1 else (120, 120, 120)
        cv2.putText(ui, "Apellido:", (canvas_w // 2 - 180, 280),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_apellido, 1)
        cv2.rectangle(ui, (canvas_w // 2 - 180, 288), (canvas_w // 2 + 180, 318), color_apellido, 1)
        cursor_a = "|" if campo_activo == 1 and parpadeo % 40 < 20 else ""
        cv2.putText(ui, apellido + cursor_a, (canvas_w // 2 - 170, 312),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # Selector de camara
        color_cam = (0, 255, 255) if campo_activo == 2 else (120, 120, 120)
        cv2.putText(ui, "Camara:", (canvas_w // 2 - 180, 350),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_cam, 1)
        cv2.rectangle(ui, (canvas_w // 2 - 180, 358), (canvas_w // 2 + 180, 358 + len(camaras) * 28 + 8), color_cam, 1)

        for i, (idx, cam_nombre) in enumerate(camaras):
            y_pos = 380 + i * 28
            if i == cam_seleccionada:
                cv2.rectangle(ui, (canvas_w // 2 - 175, y_pos - 16), (canvas_w // 2 + 175, y_pos + 8), (60, 60, 60), -1)
                marcador = ">"
                color_texto = (0, 255, 255)
            else:
                marcador = " "
                color_texto = (180, 180, 180)
            cv2.putText(ui, f"{marcador} {cam_nombre}", (canvas_w // 2 - 170, y_pos),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_texto, 1)

        # Instrucciones
        instrucciones_y = max(380 + len(camaras) * 28 + 25, 430)
        cv2.putText(ui, "TAB: Cambiar campo | Flechas: Seleccionar camara",
                     (canvas_w // 2 - 230, instrucciones_y),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1)
        cv2.putText(ui, "ENTER: Continuar | Q: Salir",
                     (canvas_w // 2 - 230, instrucciones_y + 20),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1)

        listo = nombre and apellido and camaras
        if listo and campo_activo == 2:
            cv2.putText(ui, ">> Presiona ENTER para iniciar el analisis <<",
                         (canvas_w // 2 - 200, instrucciones_y + 50),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

        cv2.imshow("Asistente Emocional", ui)
        parpadeo += 1

        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            return None, None, None
        elif key == 9:  # TAB
            campo_activo = (campo_activo + 1) % 3
        elif key == 13:  # ENTER
            if campo_activo == 0 and nombre:
                campo_activo = 1
            elif campo_activo == 1 and apellido:
                campo_activo = 2
            elif campo_activo == 2 and listo:
                return nombre, apellido, camaras[cam_seleccionada][0]
        elif key == 8 or key == 127:  # Backspace
            if campo_activo == 0:
                nombre = nombre[:-1]
            elif campo_activo == 1:
                apellido = apellido[:-1]
        elif campo_activo == 2 and camaras:
            if key == 82 or key == 0:  # Flecha arriba
                cam_seleccionada = (cam_seleccionada - 1) % len(camaras)
            elif key == 84 or key == 1:  # Flecha abajo
                cam_seleccionada = (cam_seleccionada + 1) % len(camaras)
        elif 32 <= key <= 126:  # Caracteres imprimibles
            if campo_activo == 0 and len(nombre) < 25:
                nombre += chr(key)
            elif campo_activo == 1 and len(apellido) < 25:
                apellido += chr(key)


try:
    print("Detectando camaras disponibles...")
    camaras_disponibles = detectar_camaras()
    if not camaras_disponibles:
        raise RuntimeError("No se encontraron camaras disponibles")
    print(f"Se encontraron {len(camaras_disponibles)} camara(s)")

    nombre_usuario, apellido_usuario, cam_index = pantalla_inicio(camaras_disponibles)
    if nombre_usuario is None:
        raise KeyboardInterrupt

    cam = cv2.VideoCapture(cam_index)
    if not cam.isOpened():
        raise RuntimeError(f"No se pudo abrir la camara {cam_index}")

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
        debug_metrics = {}

        if res.multi_face_landmarks:
            for face in res.multi_face_landmarks:
                # Puntos de referencia
                ojo_izq, ojo_der = get_punto(face, 133), get_punto(face, 362)
                dist_ojos = np.linalg.norm(ojo_izq - ojo_der) # Nuestra "regla" universal

                # Evitar división por cero si los landmarks están mal detectados
                if dist_ojos < 1.0:
                    continue

                # Dibujar malla facial por zonas
                dibujar_rostro(ui, face)

                # --- MÉTRICAS RELATIVAS A LA DISTANCIA DE OJOS ---
                # Boca
                apertura_boca = np.linalg.norm(get_punto(face,13) - get_punto(face,14)) / dist_ojos
                ancho_boca = np.linalg.norm(get_punto(face,61) - get_punto(face,291)) / dist_ojos

                # Cejas (Enojo) - distancia entre cejas interiores
                # Cuando frunces el ceño, las cejas se acercan entre sí
                ceja_interior_izq = get_punto(face, 55)
                ceja_interior_der = get_punto(face, 285)
                dist_cejas = np.linalg.norm(ceja_interior_izq - ceja_interior_der) / dist_ojos

                # Labios (Comisuras para sonrisa o tristeza)
                comisura_izq = get_punto(face,61)
                comisura_der = get_punto(face,291)
                centro_boca = get_punto(face,0)
                # Si las comisuras están más arriba que el centro (Y es menor), es sonrisa
                curvatura = (centro_boca[1] - (comisura_izq[1] + comisura_der[1]) / 2) / dist_ojos

                # Guardar métricas para debug
                debug_metrics = {
                    "dist_cejas": dist_cejas,
                    "curvatura": curvatura,
                    "boca": apertura_boca
                }

                # --- REGLAS DE DETECCIÓN ---
                if apertura_boca > UMBRAL_SORPRESA_BOCA:
                    emocion_detectada = "SORPRESA"
                elif dist_cejas < UMBRAL_ENOJADO_CEJAS:
                    emocion_detectada = "ENOJADO"
                elif curvatura > UMBRAL_FELIZ_CURVATURA:
                    emocion_detectada = "FELIZ"
                elif curvatura < UMBRAL_TRISTE_CURVATURA:
                    emocion_detectada = "TRISTE"
                else:
                    emocion_detectada = "NEUTRO"

                break  # Solo procesar la primera cara detectada

        buffer_emociones.append(emocion_detectada)
        final_emo = obtener_emocion(list(buffer_emociones))

        # Registrar emoción solo cuando cambia
        if final_emo != ultima_emocion:
            log_emociones.append((datetime.now().strftime("%Y-%m-%d %H:%M:%S"), final_emo))
            ultima_emocion = final_emo

        # --- PANEL DERECHO ---
        textos = {
            "FELIZ": ["SONRISA DETECTADA", "Te ves genial.", "Sigue con esa actitud!"],
            "TRISTE": ["ESTADO: TRISTE", "Pareces decaido.", "Quieres hablar con alguien?"],
            "ENOJADO": ["ESTADO: ENOJADO", "Relaja el rostro.", "Respira y cuenta hasta 10."],
            "SORPRESA": ["¡OH! SORPRESA", "Viste algo increible?", "Respira con calma."],
            "NEUTRO": ["MODO: NEUTRO", "Rostro relajado.", "Todo fluye con calma."]
        }

        info = textos[final_emo]
        cv2.putText(ui, f"{nombre_usuario} {apellido_usuario}", (ANCHO_CAM + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(ui, info[0], (ANCHO_CAM + 20, 70), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(ui, info[1], (ANCHO_CAM + 20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(ui, info[2], (ANCHO_CAM + 20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Mostrar valores de debug
        if DEBUG and debug_metrics:
            cv2.putText(ui, "--- DEBUG ---", (ANCHO_CAM + 20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
            cv2.putText(ui, f"Dist cejas: {debug_metrics['dist_cejas']:.3f} (< {UMBRAL_ENOJADO_CEJAS} = enojo)", (ANCHO_CAM + 20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 255), 1)
            cv2.putText(ui, f"Curvatura: {debug_metrics['curvatura']:.3f} (umbral: {UMBRAL_FELIZ_CURVATURA})", (ANCHO_CAM + 20, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 255, 150), 1)
            cv2.putText(ui, f"Boca: {debug_metrics['boca']:.3f} (umbral: {UMBRAL_SORPRESA_BOCA})", (ANCHO_CAM + 20, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 150, 150), 1)

        # Instrucción para guardar sesión
        cv2.putText(ui, "Ctrl+G: Guardar sesion", (ANCHO_CAM + 20, ALTO_CAM - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
        cv2.putText(ui, "Q: Salir", (ANCHO_CAM + 20, ALTO_CAM - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

        # Mostrar mensaje de guardado temporal
        if msg_guardado_timer > 0:
            cv2.putText(ui, "Sesion guardada en:", (ANCHO_CAM + 20, ALTO_CAM - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            # Partir la ruta en líneas si es muy larga
            ruta = msg_guardado
            max_chars = 30
            y_offset = ALTO_CAM - 70
            for i in range(0, len(ruta), max_chars):
                cv2.putText(ui, ruta[i:i+max_chars], (ANCHO_CAM + 20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                y_offset += 15
            msg_guardado_timer -= 1

        cv2.imshow("Asistente Emocional", ui)
        key = cv2.waitKey(FRAME_DELAY) & 0xFF
        if key == ord('q'):
            break
        elif key == 7:  # Ctrl+G
            ahora = datetime.now()
            nombre_archivo = ahora.strftime("%Y%m%d-%H%M%S") + "-log.txt"
            ruta_archivo = os.path.join(SCRIPT_DIR, nombre_archivo)
            titulo_sesion = f"Analisis de emociones de {nombre_usuario} {apellido_usuario}"
            with open(ruta_archivo, "w", encoding="utf-8") as f:
                f.write(f"{titulo_sesion}\n")
                f.write(f"Fecha: {ahora.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total registros: {len(log_emociones)}\n")
                f.write("=" * 40 + "\n")
                for ts, emo in log_emociones:
                    f.write(f"{ts}  {emo}\n")
            # Generar gráfico de barras
            nombre_grafico = ahora.strftime("%Y%m%d-%H%M%S") + "-log.png"
            ruta_grafico = os.path.join(SCRIPT_DIR, nombre_grafico)
            todas_emociones = ["FELIZ", "TRISTE", "ENOJADO", "SORPRESA", "NEUTRO"]
            conteo = Counter(emo for _, emo in log_emociones)
            valores = [conteo.get(e, 0) for e in todas_emociones]
            colores = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#95a5a6"]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(todas_emociones, valores, color=colores)
            ax.set_title(f"{titulo_sesion} - {ahora.strftime('%Y-%m-%d %H:%M:%S')}")
            ax.set_ylabel("Transiciones")
            ax.set_xlabel("Emocion")
            for i, v in enumerate(valores):
                ax.text(i, v + 0.2, str(v), ha="center", fontweight="bold")
            fig.tight_layout()
            fig.savefig(ruta_grafico, dpi=150)
            plt.close(fig)

            msg_guardado = ruta_archivo
            msg_guardado_timer = FPS_OBJETIVO * 5  # Mostrar por 5 segundos
            print(f"Log guardado en: {ruta_archivo}")
            print(f"Grafico guardado en: {ruta_grafico}")

except KeyboardInterrupt:
    print("\nPrograma interrumpido por el usuario")
except RuntimeError as e:
    print(f"Error de ejecución: {e}")
except Exception as e:
    print(f"Error inesperado: {e}")
finally:
    if 'cam' in dir():
        cam.release()
    cv2.destroyAllWindows()