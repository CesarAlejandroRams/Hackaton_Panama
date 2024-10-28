from flask import Flask, request, jsonify
from flask_cors import CORS
from sentinelhub import SHConfig
import base64
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from PIL import Image
from io import BytesIO
import imageio
from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    MimeType,
    SentinelHubRequest,
    bbox_to_dimensions,
)

from threading import Thread
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os

config = SHConfig()

app = Flask(__name__)
CORS(app)



def get_coordinates(data):
    return data.get('coordenadas', [])


#Obtencion de imagenes por filtro (.gif y .jpg)
def get_true_color(coordenadas):

    aoi = (coordenadas[0], coordenadas[1], coordenadas[2], coordenadas[3])
    resolution = 60
    aoi_bbox = BBox(bbox=aoi, crs=CRS.WGS84)
    aoi_size = bbox_to_dimensions(aoi_bbox, resolution=resolution)

    fecha_inicio = '2024-01-01' 
    fecha_fin = '2024-12-01'

    fecha_inicio_dt = datetime.strptime(fecha_inicio, '%Y-%m-%d')
    fecha_fin_dt = datetime.strptime(fecha_fin, '%Y-%m-%d')
    fechas = [] 

    while fecha_inicio_dt <= fecha_fin_dt:
        fechas.append(fecha_inicio_dt.strftime('%Y-%m-%d'))
        fecha_inicio_dt += relativedelta(months=1)

    images_true_color = []

    for date in fechas:
        try:
            request = SentinelHubRequest(
                evalscript="""
                //VERSION=3

                function setup() {
                return {
                    input: ["B04", "B03", "B02", "dataMask"],
                    output: { bands: 4 }
                };
                }

                // Contrast enhance / highlight compress


                const maxR = 3.0; // max reflectance

                const midR = 0.13;
                const sat = 1.2;
                const gamma = 1.8;

                function evaluatePixel(smp) {
                const rgbLin = satEnh(sAdj(smp.B04), sAdj(smp.B03), sAdj(smp.B02));
                return [sRGB(rgbLin[0]), sRGB(rgbLin[1]), sRGB(rgbLin[2]), smp.dataMask];
                }

                function sAdj(a) {
                return adjGamma(adj(a, midR, 1, maxR));
                }

                const gOff = 0.01;
                const gOffPow = Math.pow(gOff, gamma);
                const gOffRange = Math.pow(1 + gOff, gamma) - gOffPow;

                function adjGamma(b) {
                return (Math.pow((b + gOff), gamma) - gOffPow) / gOffRange;
                }

                // Saturation enhancement

                function satEnh(r, g, b) {
                const avgS = (r + g + b) / 3.0 * (1 - sat);
                return [clip(avgS + r * sat), clip(avgS + g * sat), clip(avgS + b * sat)];
                }

                function clip(s) {
                return s < 0 ? 0 : s > 1 ? 1 : s;
                }

                //contrast enhancement with highlight compression

                function adj(a, tx, ty, maxC) {
                var ar = clip(a / maxC, 0, 1);
                return ar * (ar * (tx / maxC + ty - 1) - ty) / (ar * (2 * tx / maxC - 1) - tx / maxC);
                }

                const sRGB = (c) => c <= 0.0031308 ? (12.92 * c) : (1.055 * Math.pow(c, 0.41666666666)-0.055);
                """,
                input_data=[SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=(date, date)
                )],
                responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                bbox=aoi_bbox,
                size=aoi_size,
                config=config
            )
            response = request.get_data()
            if response:
                img = response[0]

                if np.all(img <= 10):
                    print(f"La imagen para la fecha {date} es negra. Se omite.")
                    continue
                print(f"Imagen TrueColor descargada para la fecha: {date}") 
                images_true_color.append(img)
        except Exception as e:
            print(f"Error al procesar la fecha {date}: {e}")
    
    if images_true_color:
        gif_path = 'sentinel_tc_timelapse.gif'
        
        pil_images = [Image.fromarray(img) for img in images_true_color]

        pil_images[0].save(gif_path, save_all=True, append_images=pil_images[1:], duration=500, loop=0)
        print("GIF TC creado con éxito.")

        last_image = images_true_color[-1] 
        last_image_pil = Image.fromarray(last_image)
        last_image_path = 'sentinel_true_color_last_image.png'
        last_image_pil.save(last_image_path)  
        print(f"Última imagen tc guardada como: {last_image_path}")

        with open(gif_path, "rb") as gif_file:
            img_tc_base64 = base64.b64encode(gif_file.read()).decode('utf-8')

        return img_tc_base64 
    else:
        print("No se descargaron imágenes tc.")
        return None
    
def get_NDWI(coordenadas):
    aoi = (coordenadas[0], coordenadas[1], coordenadas[2], coordenadas[3])
    resolution = 60 
    aoi_bbox = BBox(bbox=aoi, crs=CRS.WGS84)
    aoi_size = bbox_to_dimensions(aoi_bbox, resolution=resolution)

    fecha_inicio = '2024-01-01'
    fecha_fin = '2024-12-01'

    fecha_inicio_dt = datetime.strptime(fecha_inicio, '%Y-%m-%d')
    fecha_fin_dt = datetime.strptime(fecha_fin, '%Y-%m-%d')
    fechas = [] 

    while fecha_inicio_dt <= fecha_fin_dt:
        fechas.append(fecha_inicio_dt.strftime('%Y-%m-%d'))
        fecha_inicio_dt += relativedelta(months=1) 

    images_ndwi = []

    for date in fechas:
        try:
            request = SentinelHubRequest(
                evalscript=""" 
                // NDWI calculation with cloud masking
                function setup() {
                    return {
                        input: ["B03", "B08", "dataMask"],
                        output: { bands: 3 } 
                    };
                }

                const ramp = [  
                    [-0.8, 0x008000],  // Verde 
                    [0, 0xFFFFFF],     // Blanco
                    [0.8, 0x0000CC]    // Azul
                ];

                let viz = new ColorRampVisualizer(ramp); 

                function evaluatePixel(samples) {
                    
                    if (samples.dataMask === 0) {
                        return [0, 0, 0]; 
                    }
                    
                    const val = (samples.B03 - samples.B08) / (samples.B03 + samples.B08);
                    let imgVals = viz.process(val); 
                    return imgVals.concat(samples.dataMask);
                }
                """,
                input_data=[SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A,
                        time_interval=(date, date)
                    )
                ],
                responses=[ SentinelHubRequest.output_response('default', MimeType.TIFF)
                ],
                bbox=aoi_bbox, 
                size=aoi_size, 
                config=config  
            )

            response = request.get_data()
            if response:
                img = response[0]

                if np.all(img <= 10): 
                    print(f"La imagen para la fecha {date} es negra. Se omite.")
                    continue
                print(f"Imagen NDWI descargada para la fecha: {date}") 
                images_ndwi.append(img) 
            else:
                print(f"No se encontraron datos NDWI para la fecha {date}.")
        except Exception as e:
            print(f"Error al procesar la fecha {date}: {e}")

    if images_ndwi:
        gif_path = 'sentinel_ndwi_timelapse.gif'
        
        pil_images = [Image.fromarray(img) for img in images_ndwi]

        pil_images[0].save(gif_path, save_all=True, append_images=pil_images[1:], duration=500, loop=0)
        print("GIF NDWI creado con éxito.")


        last_image = images_ndwi[-1] 
        last_image_pil = Image.fromarray(last_image)
        last_image_path = 'sentinel_ndwi_last_image.png'
        last_image_pil.save(last_image_path)  
        print(f"Última imagen ndwi guardada como: {last_image_path}")


        with open(gif_path, "rb") as gif_file:
            img_ndwi_base64 = base64.b64encode(gif_file.read()).decode('utf-8')

        return img_ndwi_base64 
    else:
        print("No se descargaron imágenes NDWI.")
        return None

def get_NDVI(coordenadas):
    aoi = (coordenadas[0], coordenadas[1], coordenadas[2], coordenadas[3])
    resolution = 60 
    aoi_bbox = BBox(bbox=aoi, crs=CRS.WGS84)
    aoi_size = bbox_to_dimensions(aoi_bbox, resolution=resolution)

    fecha_inicio = '2024-01-01'
    fecha_fin = '2024-12-01'

    fecha_inicio_dt = datetime.strptime(fecha_inicio, '%Y-%m-%d')
    fecha_fin_dt = datetime.strptime(fecha_fin, '%Y-%m-%d')
    fechas = [] 

    while fecha_inicio_dt <= fecha_fin_dt:
        fechas.append(fecha_inicio_dt.strftime('%Y-%m-%d'))
        fecha_inicio_dt += relativedelta(months=1) 

    images_ndvi = []

    for date in fechas:
        try:
            request = SentinelHubRequest(
                evalscript=""" 
            //VERSION=3

            function setup() {
            return {
                input: ["B04", "B08", "dataMask"],
                output: { bands: 4 }
            };
            }

            const ramp = [
            [-0.5, 0x0c0c0c],
            [-0.2, 0xbfbfbf],
            [-0.1, 0xdbdbdb],
            [0, 0xeaeaea],
            [0.025, 0xfff9cc],
            [0.05, 0xede8b5],
            [0.075, 0xddd89b],
            [0.1, 0xccc682],
            [0.125, 0xbcb76b],
            [0.15, 0xafc160],
            [0.175, 0xa3cc59],
            [0.2, 0x91bf51],
            [0.25, 0x7fb247],
            [0.3, 0x70a33f],
            [0.35, 0x609635],
            [0.4, 0x4f892d],
            [0.45, 0x3f7c23],
            [0.5, 0x306d1c],
            [0.55, 0x216011],
            [0.6, 0x0f540a],
            [1, 0x004400],
            ];

            const visualizer = new ColorRampVisualizer(ramp);

            function evaluatePixel(samples) {
            let ndvi = index(samples.B08, samples.B04);
            let imgVals = visualizer.process(ndvi);
            return imgVals.concat(samples.dataMask)
            }
                """,
                input_data=[SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A,
                        time_interval=(date, date)
                    )
                ],
                responses=[ SentinelHubRequest.output_response('default', MimeType.TIFF)
                ],
                bbox=aoi_bbox, 
                size=aoi_size, 
                config=config  
            )

            response = request.get_data()
            if response:
                img = response[0]

                if np.all(img <= 10):
                    print(f"La imagen para la fecha {date} es negra. Se omite.")
                    continue
                print(f"Imagen NDVI descargada para la fecha: {date}") 
                images_ndvi.append(img)
        except Exception as e:
            print(f"Error al procesar la fecha {date}: {e}")

    if images_ndvi:
        gif_path = 'sentinel_ndvi_timelapse.gif'
        
        pil_images = [Image.fromarray(img) for img in images_ndvi]

        pil_images[0].save(gif_path, save_all=True, append_images=pil_images[1:], duration=500, loop=0)
        print("GIF NDVI creado con éxito.")

        last_image = images_ndvi[-1] 
        last_image_pil = Image.fromarray(last_image)
        last_image_path = 'sentinel_ndvi_last_image.png'
        last_image_pil.save(last_image_path)  
        print(f"Última imagen ndvi guardada como: {last_image_path}")

        with open(gif_path, "rb") as gif_file:
            img_ndvi_base64 = base64.b64encode(gif_file.read()).decode('utf-8')

        return img_ndvi_base64 
    else:
        print("No se descargaron imágenes NDVI.")
        return None
    
#Procesamiento de imagenes

def imagen_a_base64(imagen):
    _, buffer = cv2.imencode('.jpg', imagen)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jpg_as_text

# Función para procesar la imagen de NDVI y detectar áreas en azul
def procesar_ndvi(ndvi_images):
    # Procesar la primera imagen en la lista
    if not ndvi_images:
        print("No se proporcionaron imágenes NDVI.")
        return None

    ndvi_image = ndvi_images[0]  # Selecciona la primera imagen
    fondo_verde = cv2.imread('./ensenada/true_color.jpg')
    ndvi_height, ndvi_width = ndvi_image.shape[:2]

    if fondo_verde is None:
        print("Error: la imagen de fondo no se encontró en la ruta especificada.")
        return None

    # Ajustar tamaño de fondo
    fondo_verde_copy = cv2.resize(fondo_verde, (ndvi_width, ndvi_height))

    # Definición de rango de color gris
    gray_lower = np.array([0, 0, 50], np.uint8)
    gray_upper = np.array([180, 50, 200], np.uint8)

    # Conversión a espacio de color HSV
    hsv_frame = cv2.cvtColor(ndvi_image, cv2.COLOR_BGR2HSV)
    gray_mask = cv2.inRange(hsv_frame, gray_lower, gray_upper)

    # Encontrar contornos
    contours, _ = cv2.findContours(gray_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10:  # Cambia el área según sea necesario
            cv2.drawContours(fondo_verde_copy, [contour], -1, (255, 0, 0), 2)

    # Guardar la imagen procesada como PNG
    output_filename = 'img_ndvi_processed.png'
    cv2.imwrite(output_filename, fondo_verde_copy)

    # Convertir la imagen procesada a base64
    _, buffer = cv2.imencode('.png', fondo_verde_copy)
    img_ndvi_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return img_ndvi_base64

# Función para procesar la imagen de NDWI y detectar áreas en azul
def procesar_ndwi(ndwi_images):
    # Procesar la primera imagen en la lista
    if not ndwi_images:
        print("No se proporcionaron imágenes NDWI.")
        return None

    ndwi_image = ndwi_images[0]  # Selecciona la primera imagen
    fondo_verde = cv2.imread('./ensenada/true_color.jpg')
    ndvi_height, ndvi_width = ndwi_image.shape[:2]

    if fondo_verde is None:
        print("Error: la imagen de fondo no se encontró en la ruta especificada.")
        return None

    # Ajustar tamaño de fondo
    fondo_verde_copy = cv2.resize(fondo_verde, (ndvi_width, ndvi_height))

    # Procesamiento de color azul
    blue_lower = np.array([90, 50, 50], np.uint8)
    blue_upper = np.array([130, 255, 255], np.uint8)
    hsv_frame = cv2.cvtColor(ndwi_image, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv_frame, blue_lower, blue_upper)
    
    # Encontrar contornos y dibujarlos
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:
            cv2.drawContours(fondo_verde_copy, [contour], -1, (255, 0, 0), 2)

    # Guardar la imagen procesada como PNG
    output_filename = 'img_ndwi_processed.png'
    cv2.imwrite(output_filename, fondo_verde_copy)

    # Convertir la imagen procesada a base64
    _, buffer = cv2.imencode('.png', fondo_verde_copy)
    img_ndwi_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return img_ndwi_base64


# Función para procesar el GIF de NDVI y detectar áreas en rosa
def procesar_gif_ndvi(cap):
    # Lista para almacenar las imágenes procesadas
    processed_images = []
    
    # Definición de rango de color gris
    gray_lower = np.array([0, 0, 50], np.uint8)
    gray_upper = np.array([180, 50, 200], np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        fondo_verde_copy = np.copy(frame)
        # Conversión a espacio de color HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray_mask = cv2.inRange(hsv_frame, gray_lower, gray_upper)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(gray_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # Cambia el área según sea necesario
                cv2.drawContours(fondo_verde_copy, [contour], -1, (255, 105, 180), -1)

        # Agregar la imagen procesada a la lista
        processed_images.append(fondo_verde_copy)

    # Convertir a GIF y base64
    if processed_images:
        gif_path = 'processed_sentinel_ndvi_timelapse.gif'
        
        # Convertir las imágenes a formato PIL
        pil_images = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in processed_images]
        pil_images[0].save(gif_path, save_all=True, append_images=pil_images[1:], duration=500, loop=0)
        
        with open(gif_path, "rb") as gif_file:
            img_ndvi_base64 = base64.b64encode(gif_file.read()).decode('utf-8')
        
        return img_ndvi_base64
    else:
        print("No se procesaron imágenes NDVI.")
        return None 

# Función para procesar el GIF de NDWI y detectar áreas en rosa
def procesar_gif_ndwi(cap):
    # Lista para almacenar las imágenes procesadas
    processed_images = []
    
    # Definición de rango de color azul para NDWI
    blue_lower = np.array([90, 50, 50], np.uint8)
    blue_upper = np.array([130, 255, 255], np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Crear una copia del frame para dibujar los contornos
        fondo_azul_copy = np.copy(frame)
        
        # Conversión a espacio de color HSV y creación de la máscara
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv_frame, blue_lower, blue_upper)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # Cambia el área según sea necesario
                cv2.drawContours(fondo_azul_copy, [contour], -1, (255, 105, 180), -1)

        # Agregar la imagen procesada a la lista
        processed_images.append(fondo_azul_copy)

    # Convertir a GIF y base64
    if processed_images:
        gif_path = 'processed_sentinel_ndwi_timelapse.gif'
        
        # Convertir las imágenes a formato PIL
        pil_images = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in processed_images]
        pil_images[0].save(gif_path, save_all=True, append_images=pil_images[1:], duration=500, loop=0)
        
        with open(gif_path, "rb") as gif_file:
            img_ndwi_base64 = base64.b64encode(gif_file.read()).decode('utf-8')
        
        return img_ndwi_base64
    else:
        print("No se procesaron imágenes NDWI.")
        return None

#Funcion para detectar calles
def detectar_calles(imagenn_path):
        # Cargar la imagen
        imagenn = cv2.imread(imagenn_path)
        if imagenn is None:
            print("Error: No se pudo cargar la imagen.")
            return
        
        # Convertir la imagen a escala de grises
        gris = cv2.cvtColor(imagenn, cv2.COLOR_BGR2GRAY)

        # Aplicar un desenfoque gaussiano para reducir el ruido
        gris = cv2.GaussianBlur(gris, (7, 7), 0)

        # Detectar bordes usando el algoritmo Canny
        bordes = cv2.Canny(gris, 50, 150)

        # Usar una operación de dilatación para engrosar los bordes detectados
        kernel = np.ones((5, 5), np.uint8)
        dilatacion = cv2.dilate(bordes, kernel, iterations=1)

        # Crear una máscara de la imagen original donde se dibujarán las calles detectadas
        mascara_calles = np.zeros_like(imagenn)

        # Copiar las áreas detectadas en la máscara
        mascara_calles[dilatacion != 0] = [0, 255, 0]

        resultado = cv2.addWeighted(imagenn, 0.8, mascara_calles, 0.2, 0)

        calles_path = './calles.png'
        calles_pil = Image.fromarray(resultado)
        calles_pil.save(calles_path)

        return resultado

#-------------Funciones de graficas--------------------------#

# Función para cargar cualquier CSV y seleccionar columnas relevantes
def cargar_datos(ruta_csv):
    data = pd.read_csv(ruta_csv)
    fecha_col = [col for col in data.columns if 'date' in col.lower()][0]
    data[fecha_col] = pd.to_datetime(data[fecha_col])
    data.set_index(fecha_col, inplace=True)
    valor_col = [col for col in data.columns if 'mean' in col.lower()][0]
    data[valor_col] = data[valor_col].interpolate(method='linear')
    return data, valor_col

# Función para crear secuencias temporales
def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
    return np.array(sequences)

# Función para predecir el futuro
def predict_future(model, last_sequence, n_future):
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(n_future):
        prediction = model.predict(current_sequence.reshape(1, len(current_sequence), 1))
        future_predictions.append(prediction[0, 0])
        current_sequence = np.append(current_sequence[1:], prediction)
    
    return np.array(future_predictions)

# Función para convertir una gráfica a base64
def plot_to_base64(vv_data, vv_col, vv_future_predictions_rescaled, vh_data, vh_col, vh_future_predictions_rescaled):
    plt.figure(figsize=(10, 6))
    
    # Graficar VV
    plt.subplot(2, 1, 1)
    plt.plot(vv_data.index, vv_data[vv_col], label='Histórico VV', color='blue')
    future_dates_vv = pd.date_range(start=vv_data.index[-1], periods=len(vv_future_predictions_rescaled) + 1, freq='M')[1:]
    plt.plot(future_dates_vv, vv_future_predictions_rescaled, label='Predicciones VV', color='red')
    plt.title('Predicciones LSTM para VV')
    plt.xlabel('Fecha')
    plt.ylabel('Valor')
    plt.legend()
    
    # Graficar VH
    plt.subplot(2, 1, 2)
    plt.plot(vh_data.index, vh_data[vh_col], label='Histórico VH', color='green')
    future_dates_vh = pd.date_range(start=vh_data.index[-1], periods=len(vh_future_predictions_rescaled) + 1, freq='M')[1:]
    plt.plot(future_dates_vh, vh_future_predictions_rescaled, label='Predicciones VH', color='red')
    plt.title('Predicciones LSTM para VH')
    plt.xlabel('Fecha')
    plt.ylabel('Valor')
    plt.legend()
    
    plt.tight_layout()

    # Guardar la gráfica en un objeto BytesIO
    buffer = BytesIO()
    plt.savefig(buffer, format='jpg')
    buffer.seek(0)
    plt.close()

    # Convertir la imagen a base64
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_base64

# Función principal para procesar los datos y entrenar
def procesar_entrenar_predecir(ruta_vv, ruta_vh, sequence_length=30, n_future=6):
    vv_data, vv_col = cargar_datos(ruta_vv)
    vh_data, vh_col = cargar_datos(ruta_vh)
    
    scaler_vv = MinMaxScaler()
    scaler_vh = MinMaxScaler()
    
    vv_scaled = scaler_vv.fit_transform(vv_data[[vv_col]])
    vh_scaled = scaler_vh.fit_transform(vh_data[[vh_col]])
    
    vv_sequences = create_sequences(vv_scaled, sequence_length)
    vh_sequences = create_sequences(vh_scaled, sequence_length)
    
    vv_labels = vv_scaled[sequence_length:]
    vh_labels = vh_scaled[sequence_length:]
    
    # Definir modelo LSTM
    def crear_modelo():
        model = Sequential()
        model.add(LSTM(100, return_sequences=True, input_shape=(sequence_length, 1)))
        model.add(Dropout(0.2))  # Capa de Dropout para prevenir sobreajuste
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(1))
        model.compile(optimizer=Adam(), loss='mean_squared_error')
        return model
    
    # Entrenar modelo para VV
    model_vv = crear_modelo()
    model_vv.fit(vv_sequences, vv_labels, epochs=1000, batch_size=32)  # Ajuste de épocas y tamaño de batch
    
    # Entrenar modelo para VH
    model_vh = crear_modelo()
    model_vh.fit(vh_sequences, vh_labels, epochs=1000, batch_size=32)  # Ajuste de épocas y tamaño de batch
    
    # Predecir futuros valores
    last_sequence_vv = vv_scaled[-sequence_length:]
    last_sequence_vh = vh_scaled[-sequence_length:]
    
    vv_future_predictions = predict_future(model_vv, last_sequence_vv, n_future)
    vh_future_predictions = predict_future(model_vh, last_sequence_vh, n_future)
    
    # Desnormalizar las predicciones
    vv_future_predictions_rescaled = scaler_vv.inverse_transform(vv_future_predictions.reshape(-1, 1))
    vh_future_predictions_rescaled = scaler_vh.inverse_transform(vh_future_predictions.reshape(-1, 1))

    # Generar gráfico en formato base64
    img_base64 = plot_to_base64(vv_data, vv_col, vv_future_predictions_rescaled, vh_data, vh_col, vh_future_predictions_rescaled)

    return img_base64



@app.route('/receive-data', methods=['POST'])
def get_images():

    data = request.get_json()
    #coordenadas = get_coordinates(data)
    coordenadas = [-116.673813,31.827164,-116.532075,31.918833]
    
    img_ndvi_base64 = get_NDVI(coordenadas)
    img_tc_base64 = get_true_color(coordenadas)
    img_ndwi_base64 = get_NDWI(coordenadas)


    ndwi_images = [cv2.imread('./sentinel_ndwi_last_image.png')]
    ndwi_processed_base64 = procesar_ndwi(ndwi_images)

    ndvi_images = [cv2.imread('./sentinel_ndwi_last_image.png')]
    ndvi_processed_base64 = procesar_ndvi(ndvi_images)

    gif_ndvi = imageio.mimread('./sentinel_ndvi_timelapse.gif')
    ndvi_gif_processed_base64 = procesar_ndvi(gif_ndvi)


    gif_to_process_ndvi = cv2.VideoCapture('./sentinel_ndvi_timelapse.gif')
    procesar_gif_ndvi(gif_to_process_ndvi)

    #gif_to_process_ndwi = cv2.VideoCapture('./sentinel_ndwi_timelapse.gif')
    #procesar_gif_ndwi(gif_to_process_ndwi)

    calles = imagen_a_base64(detectar_calles('ensenada/true_color.jpg'))

    predicciones_base64 = procesar_entrenar_predecir(
    'C:/Users/adrian/Desktop/API_FINAL/ensenada/Sentinel-1 IW VV+VH-IW-DV-VH-LINEAR-GAMMA0-ORTHORECTIFIED-2019-10-12T00_00_00.000Z-2024-10-12T23_59_59.999Z.csv', 
    'C:/Users/adrian/Desktop/API_FINAL/ensenada/Sentinel-1 IW VV+VH-IW-DV-VV-DECIBEL-GAMMA0-ORTHORECTIFIED-2019-10-12T00_00_00.000Z-2024-10-12T23_59_59.999Z.csv'
)

    return jsonify({
        "img_ndvi": img_ndvi_base64,
        "img_tc": img_tc_base64,
        "img_ndwi": img_ndwi_base64,
        "img_process_ndwi" : ndwi_processed_base64,
        "img_process_ndvi" : ndvi_processed_base64,
        "gif_process_ndvi" : ndvi_gif_processed_base64,
        "calles" : calles,
        "predicciones" : predicciones_base64
    })


if __name__ == '__main__':
    app.run(debug=True)
