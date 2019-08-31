#Importation des modules

from keras.models import Sequential #neurones
from keras.layers import Convolution2D  # images
from keras.layers import MaxPooling2D # pooling image
from keras.layers import Flatten #applatissage des images
from keras.layers import Dense # Dense
from keras.layers import Dropout

# Initialisation du CNN

classifier = Sequential()

# Convolution
classifier.add(Convolution2D(filters=32, kernel_size=3, strides=1, input_shape=(150, 150, 3), activation= "relu")) # 32 = standart ctrl i pour voir la docu

#Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# ajout de la couche de convolution car resultat non satisfaisaant 

classifier.add(Convolution2D(filters=32, kernel_size=3, strides=1, activation= "relu")) # on enleve input_shape car il ne s'agit plus d'une couche d'entrée
# On rajoute le pooling car convolution est indisociable du pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))


#Flattening
classifier.add(Flatten())

# CCC
classifier.add(Dense(units=128, activation="relu"))
classifier.add(Dropout(0.3))
#Couche de sortie
classifier.add(Dense(units=1, activation= "sigmoid"))

#Compilation
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
#classifier.compile(optimizer="adam", loss="categorical_crossentropy"metrics=["accuracy"]) pour plus de 2 issues

#entrainement + génération de plus d'images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset\\training_set', # Path avec \\ car python = débile
        target_size=(150, 150), #car image de cette taille dans la couche convultion
        batch_size=32,
        class_mode='binary') #Binairy car que 2 issues

test_set = test_datagen.flow_from_directory(
        'dataset\\test_set',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(  # le model s'appel classifier
        training_set, # jeu d'entrainement
        steps_per_epoch=250, # 8000/32 = 250
        epochs=25,
        validation_data=test_set, # jeu de test
        validation_steps=63) # 2000/32 = 62.5 donc 63


# Premier resultat a 
# 40s 161ms/step - loss: 0.3045 - acc: 0.8704 - val_loss: 0.4953 - val_acc: 0.7970

# ajout d'une couche de convolution au dessus pour de meilleur resultat (ligne 22- 24)

#  loss: 0.2836 - acc: 0.8751 - val_loss: 0.4694 - val_acc: 0.8030
# pas satisfaitsant


# test avec notre image
#import de numpy pour lesarray et de keras pour les fonctions de preprocessing
import numpy as np
from keras.preprocessing import image

# on charge l'image
test_image = image.load_img("dataset\\single_prediction\\chat_3.jpg", target_size=(64, 64)) # de taille 64 x 64 car c'est ce que l'on fait depuis le début
test_image = image.img_to_array(test_image) # on rajoute les couleurs
test_image = np.expand_dims(test_image, axis=0) # on rajoute une dimension a l'indice 0 car il manque une dimension pour .predict
result = classifier.predict(test_image) # pour stoquer le resultat dans result
training_set.class_indices # pour avoir les indices
if result[0][0] == 1:    # on affiche
    prediction = "chien"
else:
    prediction = "chat"


# Suite a l'exercice je rajoute un classifier.add(Dropout(0.2)) (droupout entre chaque couche)
# j'ai enlevé dropout pour agrandir les images de 64 px a 150 et rajouter une couche de dropout dans l'avant dernière couche 
    # FIN























