# *** Script para comprobar el funcionamiento de una matriz numpy y el acceso a sus posiciones ***
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

#array = [[1]]
#print(array[0][0])
from tkinter import*
from tkinter import filedialog as fd 
from tkinter import Tk, Label, Button, Entry, ttk
import tkinter

class BotonPrueba():

    def __init__(self, master):
        self.master = master
        self.master.geometry("300x200+450+100")
        self.botonCargarModelo = Button(self.master, text="Cargar Modelo", command=self.cargar_modelo) # boton para cargar el modelo obtenido de la red
        self.botonCargarModelo.pack()
        self.botonSalir = Button(self.master,text="salir",command=self.Salir)
        self.botonSalir.pack()
        self.master.mainloop()

    def cargar_modelo(self):
        self.ruta_modelo = fd.askopenfilename()
        print(self.ruta_modelo)
        print(type(self.ruta_modelo))

    def Salir(self):
        self.master.destroy()



if __name__ == "__main__":
    root = Tk()
    BotonPrueba(root)