# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 
from openpyxl import Workbook

K.clear_session() #limpia el backend de keras

train_path = 'C:/Users/saulm/Documents/python/deep_learning/cnn/Coral_Reef_Disease/DATASET/TRAINING_SET'
test_path = 'C:/Users/saulm/Documents/python/deep_learning/cnn/Coral_Reef_Disease/DATASET/TEST_SET'

#print(os.getcwd())

#leyendo imagenes para comprobar que si se puede acceder a la carpeta de entrenamiento
'''enf_path = 'C:/Users/saulm/Documents/python/deep_learning/cnn/Coral_Reef_Disease/DATASET/TRAINING_SET/ENFERMOS/'
enf_training = []
#img_size = 150
cont = 0

for imgn in os.listdir(enf_path):
  cont+=1
  imagen = cv2.imread(os.path.join(enf_path, imgn))
  enf_training.append(imagen)

enf_training = np.array(enf_training, dtype=object)
print(enf_training.shape)

print(np.array(enf_training[972]).shape)
plt.figure()
#plt.imshow(enf_training[8])
plt.imshow(np.squeeze(enf_training[972]))
plt.colorbar()
plt.grid(False)
plt.show()'''


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        train_path,
        target_size=(200, 200),
        batch_size=32,
        class_mode='binary')

#print(training_set.n//training_set.batch_size)

test_set = test_datagen.flow_from_directory(
        test_path,
        target_size=(200, 200),
        batch_size=32,
        class_mode='binary')

#print(test_set.n//test_set.batch_size)

print(test_set.class_indices) # para saber que etiquetas tienen las clases
print(training_set.class_indices)

# ==== calculando los pasos ====
pasos_train = training_set.n//training_set.batch_size
pasos_val = test_set.n//test_set.batch_size


# Creando la Red
model = Sequential()

# Step 1 - Convolution
# si la imagen es > a 128 usar un kernel size > 3x3
model.add(Conv2D(filters=32, kernel_size=5, strides=(1,1), padding='same', activation='relu', input_shape=(200,200,3)))
model.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
model.add(Conv2D(filters=64, kernel_size=3, strides=(1,1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
model.add(Flatten())

# Step 4 - Full connection
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images




history = model.fit(training_set, steps_per_epoch=pasos_train, epochs=15, validation_data=test_set, validation_steps=pasos_val)

directorio ='./MODELO/'

if not os.path.exists(directorio):
    os.mkdir(directorio)

model.save('./MODELO/MODELO_V1.h5')
model.save_weights('./MODELO/PESOS_V1.h5')


history_dict = history.history
print(history_dict.keys())


acc      = history.history[     'accuracy' ]
val_acc  = history.history[ 'val_accuracy' ] #### llega a uno en la segunda epoca .....esta es la presicion FINAL!
loss     = history.history[    'loss' ]
val_loss = history.history['val_loss' ]   ###############grafica muy plana

print(acc)   #primer dato
print(val_acc) #segundo dato
print(loss)
print(val_loss) 

print("----------------- RESULTADO DEL MODELO -----------------")
print()
print()
print()
print()
print("La Presición FINAL del modelo es: ", history_dict['val_accuracy'][-1])

print("----------------- GRÁFICAS -----------------")
print()
print()
print()
print()

#loss = history.history['loss']
#val_loss = history_dropout.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'g', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

wb = Workbook()

hoja = wb.active
hoja.title = "registro del Entrenamiento"

num=0
fila=2
accuary=2
val_ac=3
val_l=4
lss = 5

for accu, vala, vals, loss in zip(acc, val_acc, val_loss, loss):
    hoja.cell(row=1, column=2, value='Accuracy')
    hoja.cell(row=1, column=3, value='Val_accuracy')
    hoja.cell(row=1, column=4, value='Val_loss')
    hoja.cell(row=1, column=5, value='Loss')
    hoja.cell(column=1, row=fila, value=num)
    hoja.cell(column=accuary, row=fila, value= accu)
    hoja.cell(column=val_ac, row=fila, value= vala)
    hoja.cell(column=val_l, row=fila, value= vals)
    hoja.cell(column=lss, row=fila, value= loss)
    fila+=1
    num+=1
wb.save('./MODELO/RESULTADOS.xlsx')