# -- coding: utf-8 --
"""
Created on Wed Dec  6 15:54:22 2023

@authors: Maria Vazquez y Mariana Reyes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn") # estilo de gráficas

#%%  Etiquetas de las actividades

LABELS = ['GD_Jab',
          'GD_Hook',
          'GD_Upper',
          'GI_Cross',
          'GI_Jab',
          'GI_Hook',
          'GI_Upper',
          'GD_Cross']

# El número de pasos dentro de un segmento de tiempo
TIME_PERIODS = 50

# Los pasos a dar de un segmento al siguiente; si este valor es igual a
# TIME_PERIODS, entonces no hay solapamiento entre los segmentos
STEP_DISTANCE = 5

# al haber solapamiento aprovechamos más los datos

#%% cargamos los datos

column_names = ['activity',
                    'timestamp',
                    'Ax-axis',
                    'Ay-axis',
                    'Az-axis',
                    'Vx-axis',
                    'Vy-axis',
                    'Vz-axis']

df = pd.read_csv("golpesBC.csv", header=None,
                     names=column_names)


print(df.info())

#%% Datos que tenemos

print(df.shape)


#%% Eliminamos entradas que contengan Nan --> ausencia de datos

df.dropna(axis=0, how='any', inplace=True) #axis=0 indica que vamos a mirar en las etiquetas de las filas, axis=1 se cargaría la columna entera

#%% Mostramos los primeros datos

print(df.head())

#%% Mostramos los últimos

print(df.tail())

#%% Visualizamos la cantidad de datos que tenemos
# de cada actividad 

actividades = df['activity'].value_counts()
plt.bar(range(len(actividades)), actividades.values)
plt.xticks(range(len(actividades)), actividades.index)

#%% visualizamos 

def dibuja_datos_aceleracion(subset, actividad):
    plt.figure(figsize=(5,7))
    plt.subplot(311)
    plt.plot(subset["Ax-axis"].values)
    plt.xlabel("Tiempo", fontsize=5)
    plt.ylabel("Acel X")
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.title(actividad)
    plt.subplot(312)
    plt.plot(subset["Ay-axis"].values)
    plt.xlabel("Tiempo", fontsize=5)
    plt.ylabel("Acel Y")
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.subplot(313)
    plt.plot(subset["Az-axis"].values)
    plt.xlabel("Tiempo", fontsize=5)
    plt.ylabel("Acel Z")
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

for actividad in np.unique(df['activity']):
    subset = df[df['activity'] == actividad][:50]
    dibuja_datos_aceleracion(subset, actividad)

#%% Codificamos la actividad de manera numérica

from sklearn import preprocessing

LABEL = 'ActivityEncoded'
# Transformar las etiquetas de String a Integer mediante LabelEncoder
le = preprocessing.LabelEncoder()

# Añadir una nueva columna al DataFrame existente con los valores codificados
df[LABEL] = le.fit_transform(df['activity'].values.ravel())

print(df.head())

#%% Normalizamos los datos
#SE LE RESTA LA MEDIA Y SE DIVIDE POR LA VARIANZA


df["Ax-axis"] = (df["Ax-axis"] - df["Ax-axis"].mean()) / df["Ax-axis"].std()
df["Ay-axis"] = (df["Ay-axis"] - df["Ay-axis"].mean()) / df["Ay-axis"].std()
df["Az-axis"] = (df["Az-axis"] - df["Az-axis"].mean()) / df["Az-axis"].std()

df["Vx-axis"] = (df["Vx-axis"] - df["Vx-axis"].mean()) / df["Vx-axis"].std()
df["Vy-axis"] = (df["Vy-axis"] - df["Vy-axis"].mean()) / df["Vy-axis"].std()
df["Vz-axis"] = (df["Vz-axis"] - df["Vz-axis"].mean()) / df["Vz-axis"].std()


#%% Representamos para ver que se ha hecho bien

plt.figure(figsize=(5,5))
plt.plot(df["Ax-axis"].values[:50])
plt.xlabel("Tiempo")
plt.ylabel("Acel X")

#%% Disión datos den entrenamiento y test
column_names1 = ['activity',
                    'timestamp',
                    'Ax-axis',
                    'Ay-axis',
                    'Az-axis',
                    'Vx-axis',
                    'Vy-axis',
                    'Vz-axis',
                    'ActivityEncoded']
df_train = pd.DataFrame(df[0:4800], columns = column_names1)
df_test = pd.DataFrame(df[4800:6000], columns = column_names1)
for it in range(6000, df.shape[0], 6000):
    df_train = pd.concat([df_train, df[it:4800+it]], ignore_index=True)
    df_test = pd.concat([df_test, df[it+4800:6000+it]], ignore_index=True)
    
print("Entrenamiento", df_train.shape)
print("Test", df_test.shape)

#%% comprobamos cual ha sido la división

print("Entrenamiento", df_train.shape[0]/df.shape[0])
print("Test", df_test.shape[0]/df.shape[0])

#%% Creamos las secuencias

from scipy import stats

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mode.html

def create_segments_and_labels(df, time_steps, step, label_name):

    # x, y, z acceleraciones
    N_FEATURES = 6
    segments = []
    labels = []
    for i in range(0, len(df) - time_steps, step):
        xsa = df['Ax-axis'].values[i: i + time_steps]
        ysa = df['Ay-axis'].values[i: i + time_steps]
        zsa = df['Az-axis'].values[i: i + time_steps]
        xsv = df['Vx-axis'].values[i: i + time_steps]
        ysv = df['Vy-axis'].values[i: i + time_steps]
        zsv = df['Vz-axis'].values[i: i + time_steps]
        # Lo etiquetamos como la actividad más frecuente (la moda)
        label = stats.mode(df[label_name][i: i + time_steps])[0]
        segments.append([xsa, ysa, zsa, xsv, ysv, zsv])
        labels.append(label)

    # Los pasamos a vector
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, time_steps, N_FEATURES)
    labels = np.asarray(labels)

    return reshaped_segments, labels

x_train, y_train = create_segments_and_labels(df_train,
                                              TIME_PERIODS,
                                              STEP_DISTANCE,
                                              LABEL)

x_test, y_test = create_segments_and_labels(df_test,
                                              TIME_PERIODS,
                                              STEP_DISTANCE,
                                              LABEL)

#%% observamos la nueva forma de los datos (60, 6)

print('x_train shape: ', x_train.shape)
print(x_train.shape[0], 'training samples')
print('y_train shape: ', y_train.shape)

#%% datos de entrada de la red neuronal

num_time_periods, num_sensors = x_train.shape[1], x_train.shape[2]
num_classes = le.classes_.size
print(list(le.classes_))

#%% transformamos los datos a flotantes

x_train = x_train.astype('float32')
#y_train = y_train.astype('float32')

x_test = x_test.astype('float32')
#y_test = y_test.astype('float32')

#%% Realizamos el one-hote econding para los datos de salida

from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
y_train_hot = cat_encoder.fit_transform(y_train.reshape(len(y_train),1))
y_train = y_train_hot.toarray()

#%% RED NEURONAL
epochs = 20
batch_size = 400

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Model
#from tensorflow.keras.layers import Flatten

model = Sequential()
model.add(Conv1D(100, 10, activation='relu', input_shape=(TIME_PERIODS, 
                                                            num_sensors)))
model.add(Conv1D(100, 10, activation='relu'))
model.add(Conv1D(160, 10, activation='relu'))
model.add(Conv1D(160, 10, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
print(model.summary())


#%% Guardamos el mejor modelo y utilizamos early stopping

callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
]

#%% determinamos la función de pérdida, optimizador y métrica de funcionamiento 

from keras.optimizers import Adam

# Definir el optimizador con una tasa de aprendizaje específica 
opt = Adam(lr=0.001)

# Compilar el modelo con el optimizador configurado
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


#%% Entrenamiento



history = model.fit(x_train,
                      y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      callbacks=callbacks_list,
                      validation_split=0.1,
                      verbose=1)

#%% Visualización entrenamiento

from sklearn.metrics import classification_report

plt.figure(figsize=(6, 4))
plt.plot(history.history['accuracy'], 'r', label='Accuracy of training data')
plt.plot(history.history['val_accuracy'], 'b', label='Accuracy of validation data')
plt.plot(history.history['loss'], 'r--', label='Loss of training data')
plt.plot(history.history['val_loss'], 'b--', label='Loss of validation data')
plt.title('Model Accuracy and Loss')
plt.ylabel('Accuracy and Loss')
plt.xlabel('Training Epoch')
plt.ylim(0)
plt.legend()
plt.show()

#%% Evaluamos el modelo en los datos de test

# actualizar dependiendo del nombre del modelo guardado
model = keras.models.load_model("best_model.06-0.11.h5") #COGER EL ULTIMO MODELO GUARDADO

y_test_hot = cat_encoder.fit_transform(y_test.reshape(len(y_test),1))
y_test = y_test_hot.toarray()

test_loss, test_acc = model.evaluate(x_test, y_test)

print("Test accuracy", test_acc)
print("Test loss", test_loss)

#%%
# Print confusion matrix for training data
y_pred_train = model.predict(x_train)
# Take the class with the highest probability from the train predictions
max_y_pred_train = np.argmax(y_pred_train, axis=1)
max_y_train = np.argmax(y_train, axis=1)
print(classification_report(max_y_train, max_y_pred_train))

#%%
import seaborn as sns
from sklearn import metrics

def show_confusion_matrix(validations, predictions):

    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap='coolwarm',
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

y_pred_test = model.predict(x_test)
# Toma la clase con la mayor probabilidad a partir de las predicciones de la prueba
max_y_pred_test = np.argmax(y_pred_test, axis=1)
max_y_test = np.argmax(y_test, axis=1)

show_confusion_matrix(max_y_test, max_y_pred_test)

print(classification_report(max_y_test, max_y_pred_test))

