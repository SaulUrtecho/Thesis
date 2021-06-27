from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog as fd
import cv2
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from tkinter import Tk, Label, Button, Entry, ttk
import tkinter


enfermo = './enfermo.jpg'
sano = './sano.jpg'
longitud, altura = 200, 200
longitud2, altura2 = 300, 300
modelo = './modelo/MODELO_V1.h5'
pesos = './modelo/PESOS_V1.h5'

with CustomObjectScope({'GlorotUniform':glorot_uniform()}):
    cnn = load_model(modelo)
cnn.load_weights(pesos)


class Principal:

    def __init__(self, master):
        self.master = master
        master.title("Predicción de Salud en Corales del Caribe Mexicano")
        self.etiqueta = Label(master, text = "Seleccione Imagen a Evaluar: ")
        self.etiqueta.pack()
        self.botonCargar = Button(master, text = "Cargar Imagen", command = self.select_image) # Cargar la imagen
        self.botonCargar.pack()
        
    def select_image(self):
        global panel_A

        self.path = fd.askopenfilename()

        if len(self.path) > 0:
            image = cv2.imread(self.path)
            image = cv2.resize(image, (longitud2, altura2)) 
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)
            self.botonPredict = Button(text="Detectar Estado de Salud", command = self.predict)
            self.botonPredict.pack()

        if panel_A is None:
            self.panel_A = Label(image = image)
            self.panel_A.image = image
            self.panel_A.pack(side = "bottom")
        else:
            self.panel_A.configure(image = image)
            self.panel_A.image = image
        root.mainloop()

    def predict(self):
        global panel_B

        x = load_img(self.path, target_size = (longitud, altura))
        x = img_to_array(x)
        x = np.expand_dims(x, axis = 0)

        answer = (cnn.predict(x) > 0.5).astype("int32") # Aqui se realiza la prediccion
        respuesta = answer[0][0]
        
        if respuesta == 0:
            print(respuesta)
            self.subVentana = Toplevel()
            self.subVentana.geometry("800x400+0+0")
            self.subVentana.wm_title("Informacion Sobre Estado de Salud")
            self.subVentana.focus_set()
            self.subVentana.grab_set()
            self.pred = Label(self.subVentana, text = "CORAL CON ENFERMEDAD PRESENTE")
            self.pred.grid(row = 0, column = 0)
            self.pred.config(fg = "blue", font = ("verdana", 12))
           
            imagen = cv2.imread(enfermo)
            imagen = cv2.resize(imagen, (longitud, altura)) 
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            imagen = Image.fromarray(imagen)
            imagen = ImageTk.PhotoImage(imagen)
            
            if panel_B == None:
                self.panel_B = Label(self.subVentana, image = imagen)
                self.panel_B.image = imagen
                self.panel_B.grid(row = 1, column = 0)

                self.text = Text(self.subVentana, width = 40, height = 10)
                self.text.insert(tkinter.END, 'El coral tiene una anomalia en su estructura')
                self.text.grid(row = 1, column = 1)

                self.botonNvaDeteccion = Button(self.master, text = "Nueva Detección", command = self.Cle)
                self.botonNvaDeteccion.pack()
                self.botonSalir = Button(self.master, text = "Salir", command = self.Salir)
                self.botonSalir.pack()


        elif respuesta == 1:
            print(respuesta)
            self.subVentana = Toplevel()
            self.subVentana.geometry("800x400+0+0")
            self.subVentana.wm_title("Informacion de Salud coralina")
            self.subVentana.focus_set()
            self.subVentana.grab_set()
            self.pred = Label(self.subVentana, text = "Coral sin anomalias visibles")
            self.pred.grid(row = 0, column = 0)
            self.pred.config(fg = "pink", font = ("verdana", 12))
           
            imagen = cv2.imread(sano)
            imagen = cv2.resize(imagen, (longitud, altura)) 
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            imagen = Image.fromarray(imagen)
            imagen = ImageTk.PhotoImage(imagen)
            

            if panel_B == None:
                self.panel_B = Label(self.subVentana, image = imagen)
                self.panel_B.image = imagen
                self.panel_B.grid(row = 1, column = 0)

                self.text = Text(self.subVentana, width = 40, height = 10)
                self.text.insert(tkinter.END, 'El coral tiene una estructura normal sin niguna alteracion')
                self.text.grid(row = 1, column = 1)

                self.botonNvaDeteccion = Button(self.master, text = "Nueva Detección", command = self.Cle)
                self.botonNvaDeteccion.pack()
                self.botonSalir = Button(self.master, text = "Salir", command = self.Salir)
                self.botonSalir.pack()


        def _quit():
            self.subVentana.quit() 
            self.subVentana.destroy() 
                                

        buttonCerrar = Button(self.subVentana, text = "Cerrar", command = _quit)
        buttonCerrar.grid(row = 2, column = 3)

        self.master.wait_window(self.subVentana)

        return respuesta

    def Cle(self):
        self.botonPredict.pack_forget()
        self.botonNvaDeteccion.pack_forget()
        self.botonSalir.pack_forget()
        self.panel_A.pack_forget()
        
    def Salir(self):
            root.destroy()
      
            
root = Tk()
panel_A = None
panel_B = None    
miVentana = Principal(root)
root.geometry("500x500+450+100")
root.mainloop()