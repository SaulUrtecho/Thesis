#ESTE SCRIPT ME SIRVIO PARA CONVERTIR LAS IMAGENES FALTANTES DE PNG A JPG
import glob
from PIL import Image
import re 
import os


imgpath = 'C:/Users/saulm/Documents/python/deep_learning/cnn/Coral_Reef_Disease/test_set'

images = []
dircount = []
cant = 0

print("leyendo imagenes de: ", imgpath)

for root, dirnames, filenames in os.walk(imgpath):
    for filename in filenames:
        if re.search("\.(jpg|jpeg|JPG|png|bmp|tiff)$", filename):
            filepath = os.path.join(root, filename)
            imagen = Image.open(filepath)

            #convirtiendo a RBG
            imagen = imagen.convert("RGB")
            imagen.save('./convertidasEnf/' + str(cant) + ".jpg")
            cant = cant + 1

            b = "Leyendo..." + str(cant)
            print(b,end="\r")
           
dircount.append(cant)

print("Imagenes en cada directorio", dircount)
print("suma total de imagenes en subdirs", sum(dircount))
