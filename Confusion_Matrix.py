import numpy as np 
import pandas as pd 
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt 

test_path = 'C:/Users/saulm/Documents/python/deep_learning/cnn/Coral_Reef_Disease/DATASET/TEST_SET'

# Cargamos el modelo

modelo = load_model('./MODELO/MODELO_V1.h5')

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(200, 200),
        batch_size=32,
        class_mode='binary',
        shuffle=False)


prediccion = (modelo.predict(test_generator, steps=len(test_generator), verbose=1)>0.5).astype('int32') # Se crea el GENERADOR con la prediccion

print('El tamaño del generador es: ', len(prediccion))
print(prediccion)

clases = np.round(prediccion)

filenames = test_generator.filenames

real = []   # recorremos las imagenes y asignamos etiquetas a las clases 0 Enfermos y 1 Sanos


for i in range(0, 250):
    real.append(0)
    if i == 249:
        for i in range(249, 499):
            real.append(1) 

print(real) # valores reales (y_true)


print()
print()
print()
print("tamaño de array 2: ", len(real))

#results = pd.DataFrame({"file":filenames, "pr":prediccion[:,0], "class":clases[:,0]})
#results.to_excel('./MODELO/Predicciones.xlsx', sheet_name='Resultados de las Predicciones')
#print(results)
results = pd.DataFrame({"file":filenames, "pr":prediccion[:,0], "class":real[:]})
results.to_excel('./MODELO/Predicciones.xlsx', sheet_name='Resultados de las Predicciones')
print(results)



print("------------------------ MATRIZ POR PANDAS---------------")

data = {'y_Actual': real, 'y_Predicted': prediccion[:,0]}

df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])

confusion_matriz = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
print(confusion_matriz)



##------------------------ MATRIZ DE SKLEARN------------------------
print("-----------------Matriz SKLEARN")
print()


cm= confusion_matrix(real, prediccion)
print(cm)
clases = ['Sick: 0', 'Healthy: 1']
report= classification_report(real, prediccion, target_names=clases)
print(report)

print("-----------------METRICAS DE MATRIZ DE CONFUSION (SKLEARN)--------------")
print()
ac = accuracy_score(real, prediccion)   #exactitud
print('Puntaje de Precision: ', ac)
rc = recall_score(real,prediccion,average=None)  #recordar
print('Puntaje de Recuperacion: ', rc)
ps = precision_score(real,prediccion,average=None) #precision
print('Puntaje de Presicion', ps)
f1 = f1_score(real,prediccion,average=None)  # puntuacion f1 medida de precision y robustez del modelo
print('Puntaje F1: ', f1)


######### Se Guardan Las Metricas en un Archivo de Texto ##########

archivoPuntajes = open('./MODELO/Scores.txt', 'w')
archivoPuntajes.write('Puntaje de Clasificación de Precisión: ' + str(ac) + '\n')
archivoPuntajes.write('\n')
archivoPuntajes.write('Puntaje de Recuperación: ' + str(rc) + '\n')
archivoPuntajes.write('\n')
archivoPuntajes.write('Puntaje de Precisión TP/(TP + FP): ' + str(ps) + '\n')
archivoPuntajes.write('\n')
archivoPuntajes.write('Puntaje F1: ' + str(f1))
archivoPuntajes.close()

print("-------------Matriz GRAFICA-----------------")

plt.clf()
plt.imshow(cm, interpolation='nearest', cmap= plt.cm.get_cmap('ocean_r'))
classNames = ['Sick','Healthy']
plt.title('Matriz de confusión set de Validación')
plt.ylabel('respuesta')
plt.xlabel('predicción')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(cm[i][j]))
plt.savefig('./MODELO/MatrizConfusion.png')
plt.show()
plt.close()