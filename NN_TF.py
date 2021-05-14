#Este codigo se basa en el tutorial desarrollado en la página oficial de TensorFlow. 
#https://www.tensorflow.org/tutorials/keras/classification?hl=en

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras # API que nos ayuda a reducir el código.
# Helper libraries
import numpy as np #Utilizado para encontrar el numero max de un array
import matplotlib.pyplot as plt #Visualizar las imagenes del dataset

print(tf.__version__)


#Para realizar el aprendizaje en ML, es necesario contar con un conjunto voluminoso de datos. 
#En este caso, utilizamos el conjunto de datos Fashion MNIST, que cuenta con 70 mil imagenes de 10 tipos diferentes de prendas 
#Dentro del conjunto, 60 mil imagenes son utilizadas para entrenar a la red neuronal
# Y 10 mil, se uzan para evaluar su precición al clasificar las imagenes.

data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()

# La imagenes cuentan con tain:labels y test_labels que indican el nombre  de cada imagen.
# Se crea una lista que guarda el nombre de las etiquetas que se utilizarán en las imagenes de acuerdo a su indice 

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Se realiza el procesamiento de las imagenes, estas son de 28 * 28 pixeles.
# y cada pixel tienen un rango de color entre 0 a 255, estos valores se esclaron 
# con la finalidad de simplificar el entrenamiento en esta red neuronal simple. 
# La escala debe ser la misma en ambos  conjuntos (de entranimiento y de prueba)
# La nueva escala es  0 y 1, que entrega imagenes en blanco y negro.

train_images = train_images/255.0 #Escalando el rango de colores
test_images = test_images/255.0  #Escalando el rango de colores

# Creando el modelo. 
#EL modelo se forma de forma secuencial.

model = keras.Sequential([

    #Aquí se encuentra la primera capa de la red,  la cual recibe los datos previamente procesados (los pixeles de la imagen )
    #Estos datos estan en un array bidimencional y la primera capa se encarga de convertirlos a un array unidimensional. (de 784 pixeles)
    #la primer capa no aprende, solo reformatea los datos.

    keras.layers.Flatten(input_shape=(28,28)), 

    #Posteriormente se crea una capa que es capaz de analisar y aprender de los datos. 
    #Esta es una capa densamente conectada y consta de 128 neuronas y con una función de activación
    #De tipo 'relu' 
    #Es importante destacar que, de esta capa depende la presición con la que la red puede clasificar
    #las prendas, entre menor sea el numero de neuronas, más probable es que la red se equivoque
    #Por otro lado, utilizar un gran numero de neuronas, puede resultar redundate.
    #tras varias pruebas, se obto por utilizar tan solo 120 neuronas

    keras.layers.Dense(120, activation='relu'),

    #La ultima capa densamente conectada, devuelve un array logistico de tamaño 10 (recordemos que tenemos 10 clasificaciones de prendas)
    #cada neurona contiene un score que indica que la imagen presente corresponte a una de las clasificaciones. 
    #Se le une una función de activación softmax que ptracticamente nos permite identificar cual es la prenda seleccionada en la predicción
    keras.layers.Dense(10, activation='softmax')
])

#Compilación del modelo
#Previo al entrenamiento de la red es necesario realizar algunos ajustes.
#La función optimizadora actualiza el modelo deacuerdo a los datos recibidos  y la loss function
#'adam' es utilizado para el procesamiento de los gradientes estocásticos.
#La fución de perdida: En base a la precisión del modelo durante su entrenamiento, esta función ayuda al modelo
#a dirigirse en la dirección correcta.
#Por último, las metricasa se usan para monitorear el entrenamiento y la prueba, en esta ocación
#se calcula la frecuencia con la que la predicción coincide con la etiqueta. 

model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])


#INICIO DEL ENTRENAMIENTO
#la función fit ajusta el modelo a los datos de entrenamiento.
#En este caso, los datos d eentrenamiento definidos son tain_images  train_labels, así mismo, 
# se define cuantas veces se realizará el entrenamiento
model.fit(train_images, train_labels, epochs=7)


# Se ajusta el numero de neuronas con las repeticiones del entrenamiento 
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Tested Accuracy is: {test_acc}')

#Se ejecuta el programa para realizar las predicciones en bade a el conjunto de test_images
#La prueba se realiza sobre las primeras 10 imagenes .
#El programa mostrara la imagen en cuestión y su etiqueta de nombre asociada en la parte inferior, 
# mientras que la predicción realizada es mostrada en la parte superiór. 
prediction =  model.predict(test_images)
for i in range(10): 
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel(f"Actual Image: {class_names[test_labels[i]]}")
    plt.title(f"Prediction: {class_names[np.argmax(prediction[i])]}")
    plt.show()