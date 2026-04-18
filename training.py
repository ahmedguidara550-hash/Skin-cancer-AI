#sur colab :
#1ére partie:
from google.colab import drive
drive.mount('/content/drive')
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf

print("Installation et connexion réussies ! ✅")

#2éme:
data_dir = '/content/drive/MyDrive/projet_IA_Peau/data'
train_dir = os.path.join(data_dir,'train')
test_dir = os.path.join(data_dir,'test')

train_datagen = ImageDataGenerator(
    rescale=1./255,           
    rotation_range=25,        
    width_shift_range=0.2,    
    height_shift_range=0.2,   
    shear_range=0.2,          
    zoom_range=0.2,           
    horizontal_flip=True,     
    fill_mode='nearest'       
)


test_datagen = ImageDataGenerator(rescale=1./255)

print("Chargement des images d'entraînement (Train) :")
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),   
    batch_size=32,            
    class_mode='binary'       
)

print("\nChargement des images de test (Test) :")
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

#3eme:
print("classes détectées : ",train_generator.class_indices)
x_batch, y_batch = next(train_generator)
plt.figure(figsize=(10,10))
for i in range(9):
  plt.subplot(3,3,i+1)
  plt.imshow(x_batch[i])
  label = "Malignant" if y_batch[i]==1 else "Benign"
  plt.title(label)
  plt.axis('off')
plt.suptitle("exemples d'images du jeu d'entrainement",fontsize=14)
plt.show()

#4eme:
base_model = VGG16(weights='imagenet',include_top=False,input_shape=(224,224,3))
for layer in base_model.layers:
  layer.trainable = False

x = Flatten()(base_model.output)
x = Dense(256,activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1,activation='sigmoid')(x)

model = Model(inputs=base_model.input,outputs=output)

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

#5eme: la partie la plus importante :
print("début de l'entrainement")
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=10
)
print("\nÉvaluation terminée. Sauvegarde en cours...")
model_path = '/content/drive/MyDrive/Projet_IA_Peau/vgg16_skin_cancer.h5'
model.save(model_path)
print("Alhamdlleh ,Modèle sauvegardé avec succés ici : {model_path}")

# partie 6: Affichage des courbes d'apprentissage (Précision et Perte)
plt.figure(figsize=(12, 4))

# Courbe de Précision (Accuracy)
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entraînement', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation (Test)', color='orange')
plt.title('Évolution de la Précision')
plt.xlabel('Époques (Epochs)')
plt.ylabel('Précision')
plt.legend()

# Courbe de Perte (Loss / Erreur)
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entraînement', color='blue')
plt.plot(history.history['val_loss'], label='Validation (Test)', color='orange')
plt.title('Évolution de la Perte (Erreur)')
plt.xlabel('Époques (Epochs)')
plt.ylabel('Perte')
plt.legend()

plt.tight_layout()
plt.show()

#  Matrice de Confusion et Rapport de Classification
print("\nCalcul des prédictions pour la matrice de confusion...")

# On remet le générateur de test à zéro pour lire les images dans le bon ordre
test_generator.reset() 

# L'IA fait ses prédictions sur les images de test
predictions = model.predict(test_generator)
# On transforme les probabilités en 0 ou 1 (Si > 0.5 alors 1/Malignant, sinon 0/Benign)
y_pred = (predictions > 0.5).astype(int).reshape(-1)
# Les vraies réponses
y_true = test_generator.classes

# Noms des classes
class_names = list(test_generator.class_indices.keys())

# Rapport détaillé
print("\n📊 Rapport de Classification :")
print(classification_report(y_true, y_pred, target_names=class_names))

# Dessin de la matrice de confusion
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Matrice de Confusion')
plt.ylabel('Vraie classe (Réalité)')
plt.xlabel('Classe prédite par l\'IA')
plt.show()
