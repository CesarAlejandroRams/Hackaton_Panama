# Importar las librerías necesarias 
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# 1. Cargar los archivos CSV
vh_data = pd.read_csv("ensenada/Sentinel-1 IW VV+VH-IW-DV-VH-LINEAR-GAMMA0-ORTHORECTIFIED-2019-10-12T00_00_00.000Z-2024-10-12T23_59_59.999Z.csv")
vv_data = pd.read_csv("ensenada/Sentinel-1 IW VV+VH-IW-DV-VV-DECIBEL-GAMMA0-ORTHORECTIFIED-2019-10-12T00_00_00.000Z-2024-10-12T23_59_59.999Z.csv")

# 2. Convertir la columna de fechas al formato datetime
vv_data['date'] = pd.to_datetime(vv_data['C0/date'])
vh_data['date'] = pd.to_datetime(vh_data['C0/date'])

# 3. Asegurarse de que ambas series temporales están ordenadas por fecha
vv_data.sort_values('date', inplace=True)
vh_data.sort_values('date', inplace=True)

# 4. Establecer la columna de fechas como índice
vv_data.set_index('date', inplace=True)
vh_data.set_index('date', inplace=True)

# 5. Interpolación de datos faltantes
vv_data['C0/mean'] = vv_data['C0/mean'].interpolate(method='linear')
vh_data['C0/mean'] = vh_data['C0/mean'].interpolate(method='linear')

# 6. Normalización de los datos utilizando escaladores separados
scaler_vv = MinMaxScaler(feature_range=(0, 1))
scaler_vh = MinMaxScaler(feature_range=(0, 1))

vv_scaled = scaler_vv.fit_transform(vv_data[['C0/mean']])
vh_scaled = scaler_vh.fit_transform(vh_data[['C0/mean']])

# 7. Crear secuencias temporales con mayor longitud
def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
    return np.array(sequences)

sequence_length = 30  # Aumentar la longitud de la secuencia

# Crear secuencias para los datos VV y VH
vv_sequences = create_sequences(vv_scaled, sequence_length)
vh_sequences = create_sequences(vh_scaled, sequence_length)

# Separar características y etiquetas
X_vv = vv_sequences
y_vv = vv_scaled[sequence_length:]  # Etiquetas desde el índice "sequence_length"

X_vh = vh_sequences
y_vh = vh_scaled[sequence_length:]

# 8. Crear un modelo LSTM ajustado
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=input_shape))  # Más unidades LSTM
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Crear los modelos para VV y VH
model_vv = create_lstm_model((X_vv.shape[1], X_vv.shape[2]))
model_vh = create_lstm_model((X_vh.shape[1], X_vh.shape[2]))

# 9. Entrenar los modelos con más épocas
epochs = 100
batch_size = 32
history_vv = model_vv.fit(X_vv, y_vv, epochs=epochs, batch_size=batch_size, validation_split=0.2)
history_vh = model_vh.fit(X_vh, y_vh, epochs=epochs, batch_size=batch_size, validation_split=0.2)

# 10. Función para realizar predicciones futuras con menor cantidad de predicciones
def predict_future(model, last_sequence, n_predictions):
    future_predictions = []
    current_sequence = last_sequence

    for _ in range(n_predictions):
        prediction = model.predict(current_sequence.reshape(1, sequence_length, 1))
        future_predictions.append(prediction[0, 0])
        # Actualizar la secuencia: eliminar el primer valor y añadir la predicción
        current_sequence = np.append(current_sequence[1:], prediction)
    
    return np.array(future_predictions)

# 11. Predecir los próximos 6 valores (menos predicciones) para VV y VH
n_future = 6
last_sequence_vv = vv_scaled[-sequence_length:]
last_sequence_vh = vh_scaled[-sequence_length:]

vv_future_predictions = predict_future(model_vv, last_sequence_vv, n_future)
vh_future_predictions = predict_future(model_vh, last_sequence_vh, n_future)

# 12. Desnormalizar correctamente usando los escaladores respectivos
vv_future_predictions_rescaled = scaler_vv.inverse_transform(vv_future_predictions.reshape(-1, 1))
vh_future_predictions_rescaled = scaler_vh.inverse_transform(vh_future_predictions.reshape(-1, 1))

# 13. Graficar los resultados de las predicciones
plt.figure(figsize=(10, 6))

# Graficar para VV
plt.subplot(2, 1, 1)
plt.plot(vv_data.index, vv_data['C0/mean'], label='VV gamma0 decibel (Histórico)', color='blue')
future_dates_vv = pd.date_range(start=vv_data.index[-1], periods=n_future + 1, freq='M')[1:]
plt.plot(future_dates_vv, vv_future_predictions_rescaled, label='Predicciones Futuras VV', color='red')
plt.title('Predicciones LSTM para VV gamma0 decibel')
plt.xlabel('Fecha')
plt.ylabel('Retrodispersión (dB)')
plt.legend()
plt.grid(True)

# Graficar para VH
plt.subplot(2, 1, 2)
plt.plot(vh_data.index, vh_data['C0/mean'], label='VH gamma0 decibel (Histórico)', color='green')
future_dates_vh = pd.date_range(start=vh_data.index[-1], periods=n_future + 1, freq='M')[1:]
plt.plot(future_dates_vh, vh_future_predictions_rescaled, label='Predicciones Futuras VH', color='red')
plt.title('Predicciones LSTM para VH gamma0 decibel')
plt.xlabel('Fecha')
plt.ylabel('Retrodispersión (dB)')
plt.legend()

plt.tight_layout()
plt.show()
