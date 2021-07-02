# *** Programa para comprobar el funcionamiento de una matriz numpy y el acceso a sus posiciones ***
# esta comprobacion me sirvio al momento de crear el excel y la matriz de confusion en el modulo
# Confusion_Matrix.py
#import numpy as np
#from keras.preprocessing.image import load_img, img_to_array
#from PIL import Image, ImageTk

#arreglo = [[1,9],[2,8],[3,7],[4,5],[5,4],[7,3],[8,2],[9,1]]

#listaNumpy = np.array(arreglo) # convertimos la lista en un arreglo numpy

#print(listaNumpy[:,1])  # modificando el segundo parametro podemos acceder a las posiciones de cada subvector de la matriz

'''longitud = 200
altura = 200

x = load_img('./sano.jpg', target_size = (longitud, altura))
x = img_to_array(x)
print(x.shape)
print(type(x))
x = np.expand_dims(x, axis = 0)
print(x.shape)
print(type(x))'''

array = [[1]]
print(array[0][0])