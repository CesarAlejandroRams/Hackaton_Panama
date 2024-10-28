import cv2
import numpy as np
from threading import Thread

# Función para procesar la imagen de NDVI y detectar áreas en azul
def procesar_ndvi(ndvi_image, fondo_verde_copy):        
    gray_lower = np.array([0, 0, 50], np.uint8)
    gray_upper = np.array([180, 50, 200], np.uint8)

    hsv_frame = cv2.cvtColor(ndvi_image, cv2.COLOR_BGR2HSV)
    gray_mask = cv2.inRange(hsv_frame, gray_lower, gray_upper)
    contours, _ = cv2.findContours(gray_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10:  # Ajustar el umbral si es necesario
            cv2.drawContours(fondo_verde_copy, [contour], -1, (255, 0, 0), 2)  # Azul

# Función para procesar la imagen de NDWI y detectar áreas en azul
def procesar_ndwi(ndwi_image, fondo_verde_copy):
    blue_lower = np.array([90, 50, 50], np.uint8)
    blue_upper = np.array([130, 255, 255], np.uint8)

    hsv_frame = cv2.cvtColor(ndwi_image, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv_frame, blue_lower, blue_upper)
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:  # Ajustar el umbral si es necesario
            cv2.drawContours(fondo_verde_copy, [contour], -1, (255, 0, 0), 2)  # Azul

# Función para procesar el GIF de NDVI y detectar áreas en rosa
def procesar_gif_ndvi(cap, fondo_verde_copy, rosa_mask):
    gray_lower = np.array([0, 0, 50], np.uint8)
    gray_upper = np.array([180, 50, 200], np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray_mask = cv2.inRange(hsv_frame, gray_lower, gray_upper)
        contours, _ = cv2.findContours(gray_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:
                cv2.drawContours(rosa_mask, [contour], -1, (255, 105, 180), -1)  # Rosa en la máscara

# Función para procesar el GIF de NDWI y detectar áreas en rosa
def procesar_gif_ndwi(cap, fondo_verde_copy, rosa_mask):
    blue_lower = np.array([90, 50, 50], np.uint8)
    blue_upper = np.array([130, 255, 255], np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv_frame, blue_lower, blue_upper)
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:
                cv2.drawContours(rosa_mask, [contour], -1, (255, 105, 180), -1)  # Rosa en la máscara

# Cargar la imagen de fondo
fondo_verde = cv2.imread('./CodigosIndependientes/ensenada/2024-04-01-00_00_2024-04-01-23_59_Sentinel-2_Quarterly_Mosaics_True_Color_Cloudless.jpg')

# Cargar las imágenes de NDVI y NDWI
ndvi_image = cv2.imread('./ensenada/Sentinel-2_L1C-917111078020849-timelapse-0042.jpg')
ndwi_image = cv2.imread("./ensenada/Sentinel-2_L1C-1466683088009321-timelapse-0042.jpg")

# Obtener dimensiones de las imágenes de NDVI y NDWI
ndvi_height, ndvi_width = ndvi_image.shape[:2]
ndwi_height, ndwi_width = ndwi_image.shape[:2]

# Asegurarse de que las dimensiones coincidan
if ndvi_height != ndwi_height or ndvi_width != ndwi_width:
    print("Las imágenes de NDVI y NDWI deben tener las mismas dimensiones.")
    exit()

# Redimensionar la imagen de fondo al tamaño de las imágenes de NDVI y NDWI
fondo_verde_copy = cv2.resize(fondo_verde, (ndvi_width, ndvi_height))

# Procesar las imágenes de NDVI y NDWI primero (dibuja en azul)
procesar_ndvi(ndvi_image, fondo_verde_copy)
procesar_ndwi(ndwi_image, fondo_verde_copy)

# Cargar los GIFs
cap_ndvi = cv2.VideoCapture("./ensenada/Sentinel-2_L1C-917111078020849-timelapse.gif")
cap_ndwi = cv2.VideoCapture("./ensenada/Sentinel-2_L1C-1557339245669505-timelapse.gif")

# Crear una máscara rosa vacía
rosa_mask = np.zeros_like(fondo_verde_copy)

# Crear threads para procesar los GIFs
thread_ndvi = Thread(target=procesar_gif_ndvi, args=(cap_ndvi, fondo_verde_copy, rosa_mask))
thread_ndwi = Thread(target=procesar_gif_ndwi, args=(cap_ndwi, fondo_verde_copy, rosa_mask))

# Iniciar los threads
thread_ndvi.start()
thread_ndwi.start()

# Mostrar la imagen de fondo con los resultados de NDVI y NDWI
while True:
    # Hacer una copia de fondo para mostrar
    imagen_a_mostrar = fondo_verde_copy.copy()

    # Aplicar el color rosa solo donde haya contornos detectados
    rosa_transparente = cv2.addWeighted(imagen_a_mostrar, 0.75, rosa_mask, 0.25, 0)

    # Mostrar la imagen con los resultados
    cv2.imshow('Mapa con Resultados de NDVI y NDWI', rosa_transparente)

    # Espera a que el usuario presione 'q' para cerrar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Detener los threads de procesamiento de imágenes
thread_ndvi.join()
thread_ndwi.join()

# Cerrar todo
cap_ndvi.release()
cap_ndwi.release()
cv2.destroyAllWindows()
