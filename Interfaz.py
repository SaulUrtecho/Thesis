# ********* PARTE 3: Interfaz grafica UI para validar el modelo *********
# se utiliza la libreria tkinter

from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog as fd  # Esta libreria sirve para abrir una ventana de dialogo
import cv2
from keras_preprocessing import image
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from tkinter import Tk, Label, Button, Entry, ttk
import tkinter

# Configuramos el alto y ancho que tendran las imagenes a utilizar en la interfaz
longitud, altura = 200, 200
ruta_logo_itm = 'C:/Users/saulm/Documents/python/deep_learning/cnn/Coral_Reef_Disease/logo.jpg'
# Esta es la clase que crea la interfaz grafica se utiliza la POO
class Principal():
    # Metodo constructor el cual recibe un objeto Tk()
    # este metodo inicializa toda la interfaz
    def __init__(self, master):
        panelLogo = None
        self.master = master
        self.master.title("(TESIS) Predicción de Salud en Corales del Caribe Mexicano")
        self.master.geometry("500x400+450+100")
        self.etiquetaModel = Label(self.master, text="Seleccione el modelo: ")
        self.etiquetaModel.pack()
        self.botonCargarModelo = Button(self.master, text="Cargar Modelo", command=self.cargar_modelo) # boton para cargar el modelo obtenido de la red
        self.botonCargarModelo.pack()
        self.etiquetaPesos = Label(self.master, text="Seleccione los pesos: ")
        self.etiquetaPesos.pack()
        self.botonCargarPesos = Button(self.master, text="Cargar Pesos", command=self.cargar_pesos) # boton para cargar los pesos obtenidos de la red
        self.botonCargarPesos.pack()
        self.etiqueta = Label(self.master, text = "Seleccione Imagen a Evaluar: ")
        self.etiqueta.pack()
        self.botonCargar = Button(self.master, text = "Cargar Imagen", command = self.select_image) # Cargar la imagen
        self.botonCargar.pack()
        logo = cv2.imread(ruta_logo_itm)
        #image = cv2.resize(logo, (longitud, altura)) # Se cambia el tamaño a 300x300
        logo = cv2.cvtColor(logo, cv2.COLOR_BGR2RGB) # Se convierte a RGB
        logo = Image.fromarray(logo) # se convierte de matriz a imagen
        logo = ImageTk.PhotoImage(logo)
        self.panelLogo = Label(self.master,image = logo)
        self.panelLogo.logo = logo
        self.panelLogo.pack(side = "bottom")


        self.master.mainloop()  # Este mainloop es el que mantiene la VENTANA PRINCIPAL FUNCIONANDO

    def cargar_modelo(self): # en esta funcion obtenemos la ruta del modelo
        self.ruta_modelo = fd.askopenfilename()
        #if self.ruta_modelo != 'C:/Users/saulm/Documents/python/deep_learning/cnn/Coral_Reef_Disease/MODELO/MODELO_V1.h5':
           # self.ventana_alerta = Toplevel()


        if len(self.ruta_modelo) >0: # si se selecciono una imagen se eliminan los widgets si no no
            self.etiquetaModel.pack_forget()
            self.botonCargarModelo.pack_forget()
        return self.ruta_modelo
    
    def cargar_pesos(self): # en esta funcion obtenemos la ruta de los pesos
        self.ruta_pesos=fd.askopenfilename()
        if len(self.ruta_pesos)>0: # si se selecciono una imagen se eliminan los widgets si no no
            self.etiquetaPesos.pack_forget()
            self.botonCargarPesos.pack_forget()
        return self.ruta_pesos
        
    def select_image(self):
        #self.etiqueta.pack_forget() # al momento de seleccionar la imagen se elimina la etiqueta que pide seleccionar imagen
        #self.botonCargar.pack_forget() # el boton cargar se elimina al seleccionar la imagen
        panel_A = None # esta variable contendra la imagen que se muestra al momento de seleccionar la imagen
        self.path = fd.askopenfilename() # Abre una ventana de dialogo para seleccionar una imagen, y devuelve la ruta de la imagen seleccionada
        
        if len(self.path) > 0:  # Si alguna imagen fue seleccionada
            self.etiqueta.pack_forget() # al momento de seleccionar la imagen se elimina la etiqueta que pide seleccionar imagen
            self.botonCargar.pack_forget() # el boton cargar se elimina al seleccionar la imagen
            self.panelLogo.pack_forget()
            image = cv2.imread(self.path) # la imagen es leida
            image = cv2.resize(image, (longitud, altura)) # Se cambia el tamaño a 300x300
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Se convierte a RGB
            image = Image.fromarray(image) # se convierte de matriz a imagen
            image = ImageTk.PhotoImage(image) # La funcion PhotoImage() crea una instancia de imagen para colocar en una label
            self.imagen_copia = image # se crea una copia de la imagen para colocarla en la ventana secundaria al momento de predecir
            self.botonPredict = Button(text="Detectar Estado de Salud", command = self.predict) # se crea el boton que hace la prediccion
            self.botonPredict.pack() # el boton es colocado en la interfaz
            self.botonSelNuevament = Button(text="Seleccionar otra imagen", command = self.select_image2)
            self.botonSelNuevament.pack()

        if panel_A is None: # Si la variable panel_A esta inicializada en nula
            self.panel_A = Label(self.master, image = image)  # se carga la imagen en la label
            self.panel_A.image = image 
            self.panel_A.pack(side = "bottom")
        else:
            self.panel_A.configure(self.master, image = image)
            self.panel_A.image = image
            self.panel_A.pack(side="bottom")
      
        self.master.mainloop() # Este mainloop se coloca aqui para mantener la VENTANA PRINCIPAL FUNCIONANDO despues de cerrar la ventana secundaria del resultado de la prediccion

    def select_image2(self): # esta funcion se creo si el usuario quiere seleccionar otra imagen y no la seleccionada
        self.panel_A.pack_forget() # se limpia el panel de la imagen anterior
        self.botonPredict.pack_forget() # se limpia el boton predecir
        self.botonSelNuevament.pack_forget() # se limpia el boton para seleccionar otra imagen 
        panel_A = None # se inicializa de nuevo el panel como nulo
        self.path = fd.askopenfilename()

        if len(self.path) > 0:  # Si alguna imagen fue seleccionada
            image = cv2.imread(self.path) # la imagen es leida
            image = cv2.resize(image, (longitud, altura)) # Se cambia el tamaño a 300x300
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Se convierte a RGB
            image = Image.fromarray(image) # se convierte de matriz a imagen
            image = ImageTk.PhotoImage(image) # La funcion PhotoImage() crea una instancia de imagen para colocar en una label
            self.imagen_copia = image # se crea una copia de la imagen para colocarla en la ventana secundaria al momento de hacer la prediccion
            self.botonPredict = Button(text="Detectar Estado de Salud", command = self.predict) # se crea el boton que hace la prediccion
            self.botonPredict.pack() # el boton es colocado en la interfaz
            self.botonSelNuevament = Button(text="Seleccionar otra imagen", command = self.select_image2) # se crea el boton para seleccionar otra imagen si lo desea
            self.botonSelNuevament.pack() # el boton se coloca en la interfaz

        if panel_A is None: # Si la variable panel_A esta inicializada en nula
            self.panel_A = Label(self.master, image = image)  # se carga la imagen en la label
            self.panel_A.image = image 
            self.panel_A.pack(side = "bottom")
        else:
            self.panel_A.configure(self.master, image = image)
            self.panel_A.image = image
            self.panel_A.pack(side="bottom")
      
        self.master.mainloop()


    def predict(self):  # esta funcion se activa al momento de darle click al boton "Detectar estado de salud"
        
        self.botonPredict.pack_forget()  # los widgets son limpiados de nuevo al hacer una prediccion ya que si no se sobreescriben en los anteriores
        self.botonSelNuevament.pack_forget()
        self.panel_A.pack_forget()

        with CustomObjectScope({'GlorotUniform':glorot_uniform()}): # cargamos el modelo y los pesos obtenidos de las rutas
            cnn = load_model(self.ruta_modelo)
        cnn.load_weights(self.ruta_pesos)

        # las variables panel son creadas para representar ese espacion donde se colocara una imagen
        panel_Ventana_secundaria = None
        # Aqui se hace la prediccion obteniendo la imagen de la ruta y cambiandole su tamaño al tamaño que se uso en el entrenamiento 200x200
        x = load_img(self.path, target_size = (longitud, altura)) 
        x = img_to_array(x) # la imagen se convierte a tipo arreglo numpy con tres canales por RGB(200,200,3)
        # luego se convierte la imagen a 4D, se agrega un 1 dimension mas en la posicion especificada axis=0, quedando
        # de la siguiente manera (1,200,200,3) en donde este uno representa el tamaño de lote, este valor representa
        # la cantidad de imagenes que agrupa para alimentar la CNN, el lote puede ser una sola imagen, sin embargo
        # todavia necesita esa dimension adicional de 1 para mostrar el tamaño de lote
        x = np.expand_dims(x, axis = 0) # 

        # Aqui se realiza la prediccion si el valor de la prediccion es mayor a el umbral 0.5 entonces la salida
        # será 1, si no la salida sera 0, esto lo devuelve en un arreglo numpy [[0]] o [[1]] dependiendo de la prediccion
        answer = (cnn.predict(x) > 0.5).astype("int32") 
        respuesta = answer[0][0] # Accedemos al valor almacenado en la matriz indexando [0][0] ya que es en 2D
        
        if respuesta == 0: # si la prediccion es 0 == Enfermo
            print(respuesta)
            self.subVentana = Toplevel() # Crea una nueva ventana secundaria para mostrar la salida
            self.subVentana.geometry("600x300+300+150") # Se establece el tamaño de la ventana secundaria
            self.subVentana.wm_title("(Tesis)Información Sobre Estado de Salud") # Se le asigna el titulo
            self.subVentana.focus_set() # Este metodo enfoca la ventana secundaria 
            self.subVentana.grab_set() # desactivamos la ventana principal 
            self.pred = Label(self.subVentana, text = " Coral ENFERMO detectado") # asignamos la etiqueta para mencionar que es un coral enfermo
            self.pred.grid(row = 0, column = 0)
            self.pred.config(fg = "black", font = ("verdana", 12))
           
            
            if panel_Ventana_secundaria == None:
                self.panel_Ventana_secundaria = Label(self.subVentana, image = self.imagen_copia) # le pasamos la imagen a la etiqueta
                self.panel_Ventana_secundaria.image = self.imagen_copia
                self.panel_Ventana_secundaria.grid(row = 1, column = 0) # posicionamos la etiqueta en la fila 1 columna 0

                self.text = Text(self.subVentana, width = 40, height = 12) # creamos un campo de texto para insertar la informacion respectiva al estado de salud del coral
                self.text.insert(tkinter.END, 'El coral tiene una anomalia en su estructura') # Texto informativo
                self.text.grid(row = 1, column = 1) # se posiciona a la derecha de la imagen en la fila 1 columna 1

                # Mientras esto sucede, se crea inmediatamente un boton en la ventana principal 
                # para realizar una nueva prediccion y tambien se crea un boton en la misma ventana principal para salir
                
                self.botonNvaDeteccion = Button(self.master, text = "Nueva Detección", command = self.Cle) # al momento de realizar una nueva prediccion, los botones se eliminan y se limpia la pantalla dejando unicamente el boton para seleccionar la imagen a evaluar
                self.botonNvaDeteccion.pack()
                self.botonSalir = Button(self.master, text = "Salir", command = self.Salir) # boton para salir
                self.botonSalir.pack()

        # Se definen los mismos pasos cuando la prediccion es 1 == SANO
        elif respuesta == 1:
            print(respuesta)
            self.subVentana = Toplevel()
            self.subVentana.geometry("600x300+300+150")
            self.subVentana.wm_title("(Tesis)Información de Sobre Estado de Salud")
            self.subVentana.focus_set()
            self.subVentana.grab_set()
            self.pred = Label(self.subVentana, text = " Coral SANO detectado")
            self.pred.grid(row = 0, column = 0)
            self.pred.config(fg = "black", font = ("verdana", 12))
            
            # de igual manera se definen las mismas funcionalidades para cuando es un coral sano
            if panel_Ventana_secundaria == None:
                self.panel_Ventana_secundaria = Label(self.subVentana, image = self.imagen_copia)
                self.panel_Ventana_secundaria.image = self.imagen_copia
                self.panel_Ventana_secundaria.grid(row = 1, column = 0)

                self.text = Text(self.subVentana, width = 40, height = 12)
                self.text.insert(tkinter.END, 'El coral tiene una estructura normal sin niguna alteracion')
                self.text.grid(row = 1, column = 1)

                self.botonNvaDeteccion = Button(self.master, text = "Nueva Detección", command = self.Cle)
                self.botonNvaDeteccion.pack()
                self.botonSalir = Button(self.master, text = "Salir", command = self.Salir)
                self.botonSalir.pack()
        self.botonSelNuevament.pack_forget()


        def CerrarVentanaSecundaria(): # funcion local(pertenece a predict()) para cerrar la ventana secundaria
            self.subVentana.quit() 
            self.subVentana.destroy() 
                                
        # Se crea el boton que aparecera en la ventana secuendaria para salir
        buttonCerrar = Button(self.subVentana, text = "Cerrar", command = CerrarVentanaSecundaria)
        buttonCerrar.grid(row = 2, column = 3) # este ira en la fila 2 y columna 3

        self.master.wait_window(self.subVentana) # espera hasta que la subventana sea destruida

        return respuesta # Retorna el valor 0 o 1 dependiendo de la prediccion

    # esta funcion sirve para eliminar los widgets una vez realizada una nueva prediccion y agregar los widgets para una nueva deteccion
    def Cle(self): 
        self.botonPredict.pack_forget()
        self.botonNvaDeteccion.pack_forget()
        self.botonSalir.pack_forget()
        
        self.etiqueta.pack()
        self.botonCargar.pack()
        self.panelLogo.pack(side = "bottom")

    # esta funcion sirve para cerrar LA VENTANA PRINCIPAL   
    def Salir(self):
        self.master.destroy()
        
      
# METODO MAIN PARA HACER FUNCIONAR LA UI
if __name__ == "__main__":
    root = Tk()
    Principal(root)
  