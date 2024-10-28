import pandas as pd
import cv2
import time
import numpy as np
import tkinter as tk
from tkinter import messagebox
import threading
import matplotlib.pyplot as plt
from twilio.rest import Client

# Función para evaluar el riesgo de monóxido de carbono en el aire
def evaluate_co_risk(co_value):
    if co_value > 0.1:
        return "ALERTA: Alta concentración de CO"
    elif co_value > 0.05:
        return "Precaución: Concentración moderada de CO"
    else:
        return "Aire limpio: Niveles seguros de CO"


# Función para mostrar cuadro emergente de advertencia
def show_warning():
    root = tk.Tk()
    root.withdraw()  # Ocultar la ventana principal
    messagebox.showwarning("Advertencia", "¡Incendio inminente detectado!")
    root.destroy()

# Función para cargar y evaluar los datos de CO y mostrar el gráfico
def monitor_co():
    # Cargar los datos del archivo CSV
    data = pd.read_csv('./CodigosIndependientes/incendiosAD/Sentinel-5P.csv')

    # Asegurarse de que las columnas son correctas
    dates = pd.to_datetime(data['C0/date'])  # Ajustar si es necesario
    co_values = data['C0/mean']               # Ajustar si es necesario

    # Evaluar el riesgo de CO para cada valor
    alerts = [evaluate_co_risk(value) for value in co_values]

    # Mostrar las alertas generadas en cada fecha
    for date, value, alert in zip(dates, co_values, alerts):
        print(f"Fecha: {date}, CO: {value} mol/m², Estado: {alert}")

    # Crear el gráfico de concentración de CO
    plt.figure(figsize=(10, 6))

    # Graficar los valores de CO con sus fechas
    plt.plot(dates, co_values, marker='o', label='Concentración promedio de CO (mol/m²)', color='b')

    # Agregar líneas horizontales para los niveles de alerta y concentración moderada
    plt.axhline(y=0.1, color='r', linestyle='--', label='Alerta de alta concentración (0.1 mol/m²)')
    plt.axhline(y=0.05, color='y', linestyle='--', label='Nivel de preocupacion (0.05 mol/m²)')
    plt.axhline(y=0.0, color='g', linestyle='--', label='Nivel normal (0.0 mol/m²)')

    # Etiquetas y título del gráfico
    plt.xlabel('Fecha')
    plt.ylabel('Concentración de CO (mol/m²)')
    plt.title('Monitoreo de riesgo por concentración de monóxido de carbono')
    plt.legend()  # Mostrar la leyenda del gráfico
    plt.grid(True)  # Activar la cuadrícula
    plt.xticks(rotation=45)  # Rotar las etiquetas de las fechas para mejor lectura

    # Mostrar el gráfico
    plt.tight_layout()  # Ajustar el layout para evitar solapamientos
    plt.show()


# Función para detectar incendios
def detect_fire():
    # Cargar el GIF
    cap = cv2.VideoCapture("./CodigosIndependientes/incendiosAD/calon.gif")



    # Definir los rangos de colores en HSV para amarillo, rojo vivo y naranja
    yellow_lower = np.array([20, 100, 100], np.uint8)
    yellow_upper = np.array([30, 255, 255], np.uint8)
    
    red_vivid_lower1 = np.array([0, 150, 150], np.uint8)
    red_vivid_upper1 = np.array([10, 255, 255], np.uint8)
    red_vivid_lower2 = np.array([170, 150, 150], np.uint8)
    red_vivid_upper2 = np.array([180, 255, 255], np.uint8)
    
    orange_lower = np.array([10, 100, 100], np.uint8)
    orange_upper = np.array([20, 255, 255], np.uint8)

    areas_list = []  # Lista para almacenar las áreas detectadas
    fire_warning_shown = False  # Controla si ya se mostró la advertencia

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Análisis del GIF completado.")
            break  # Termina al final del GIF

        time.sleep(1)  # Delay de 1 segundo entre frames

        # Convertir el frame a espacio de color HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Crear máscaras para amarillo, rojo vivo y naranja
        yellow_mask = cv2.inRange(hsv_frame, yellow_lower, yellow_upper)
        red_mask1 = cv2.inRange(hsv_frame, red_vivid_lower1, red_vivid_upper1)
        red_mask2 = cv2.inRange(hsv_frame, red_vivid_lower2, red_vivid_upper2)
        orange_mask = cv2.inRange(hsv_frame, orange_lower, orange_upper)

        # Unir todas las máscaras
        combined_mask = yellow_mask + red_mask1 + red_mask2 + orange_mask

        # Encontrar contornos de las áreas detectadas
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 400:  # Filtrar objetos pequeños
                if not fire_warning_shown:
                    show_warning()  # Mostrar advertencia una vez
                    fire_warning_shown = True

                # Obtener las dimensiones del rectángulo envolvente
                x, y, w, h = cv2.boundingRect(contour)
                rect_area = w * h
                areas_list.append(rect_area)

                # Dibujar el rectángulo y mostrar información en el frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Width: {w} px", (x, y - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Height: {h} px", (x, y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Area: {rect_area} px^2", (x, y + h + 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Mostrar el frame con las detecciones
        cv2.imshow("Heat Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Salir con 'q'

    # Calcular el promedio de las áreas detectadas
    if areas_list:
        avg_area = sum(areas_list) / len(areas_list)
        print(f"\nRiesgo Relativo: {avg_area:.2f} px^2")
        with open("riesgo_relativo.txt", "w") as file:
            file.write(str(avg_area))
    else:
        print("No se detectaron áreas significativas.")

    # Liberar recursos y cerrar ventanas
    cap.release()
    cv2.destroyAllWindows()

# Crear y ejecutar hilos para las funciones
if __name__ == "__main__":
    # Hilo para monitorear el CO
    co_thread = threading.Thread(target=monitor_co)
    co_thread.start()

    # Hilo para detectar incendios
    fire_thread = threading.Thread(target=detect_fire)
    fire_thread.start()

    # Esperar a que ambos hilos terminen
    co_thread.join()
    fire_thread.join()
