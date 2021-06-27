#ESTE SCRIPT ME SIRVIO PARA ROTAR LAS IMAGENES Y ASI AUMENTAR EL DATASET
from PIL import Image
import os
import glob
import re

#print(os.getcwd())

dirname = os.path.join(os.getcwd(), './healthy')    # se une la ruta actual con la ruta de las imagenes
imgpath = dirname + os.sep      # se obtiene la ruta definitiva 

images = []
dircount = []
cant = 0

print("leyendo imagenes de: ", imgpath)

for root, dirnames, filenames in os.walk(imgpath):
    for filename in filenames:
        if re.search("\.(jpg|jpeg|JPG|png|bmp|tiff)$", filename):
            filepath = os.path.join(root, filename)
            imagen = Image.open(filepath)
            imagen_rotada90 = imagen.rotate(90)
            imagen_rotada180 = imagen.rotate(180)
            imagen_rotada270 = imagen.rotate(270)

            #convirtiendo a RBG
            imagen_rotada90 = imagen_rotada90.convert("RGB")
            imagen_rotada180 = imagen_rotada180.convert("RGB")
            imagen_rotada270 = imagen_rotada270.convert("RGB")

            imagen_rotada90.save('./rotadas_san/' + str(cant) + "_90_grados_.jpg")
            imagen_rotada180.save('./rotadas_san/' + str(cant) + "_180_grados_.jpg")
            imagen_rotada270.save('./rotadas_san/' + str(cant) + "_270_grados_.jpg")
            cant = cant + 1

            b = "Leyendo..." + str(cant)
            print(b,end="\r")
           
dircount.append(cant)
print("Imagenes en cada directorio", dircount)
print("suma total de imagenes en subdirs", sum(dircount))
