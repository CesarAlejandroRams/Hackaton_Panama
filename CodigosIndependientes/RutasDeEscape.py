import cv2
import numpy as np

def detectar_calles(imagen_path):
    # Cargar la imagen
    imagen = cv2.imread(imagen_path)
    if imagen is None:
        print("Error: No se pudo cargar la imagen.")
        return
    
    # Convertir la imagen a escala de grises
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Aplicar un desenfoque gaussiano para reducir el ruido
    gris = cv2.GaussianBlur(gris, (7, 7), 0)

    # Detectar bordes usando el algoritmo Canny
    bordes = cv2.Canny(gris, 50, 150)

    # Usar una operación de dilatación para engrosar los bordes detectados
    kernel = np.ones((5, 5), np.uint8)
    dilatacion = cv2.dilate(bordes, kernel, iterations=1)

    # Crear una máscara de la imagen original donde se dibujarán las calles detectadas
    mascara_calles = np.zeros_like(imagen)

    # Copiar las áreas detectadas en la máscara
    mascara_calles[dilatacion != 0] = [0, 255, 0]  # Verde para las calles

    # Combinar la imagen original con la máscara de calles
    resultado = cv2.addWeighted(imagen, 0.8, mascara_calles, 0.2, 0)

    # Mostrar solo la ventana con las calles detectadas
    cv2.imshow('Calles con posible peligro', resultado)

    # Esperar a que el usuario presione una tecla y cerrar las ventanas
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Ruta de la imagen que deseas procesar
ruta_imagen = 'ensenada/2024-04-01-00_00_2024-04-01-23_59_Sentinel-2_Quarterly_Mosaics_True_Color_Cloudless.jpg'
detectar_calles(ruta_imagen)
