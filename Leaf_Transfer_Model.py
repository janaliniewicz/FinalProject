import os
from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3

local_weights_file = r'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'    # Load the pre-trained weights

pre_trained_model = InceptionV3(input_shape = (150, 150,3),              #Create an instance of a model with inception's model structure
                                include_top = False,                    #Do not include the top layer
                                weights = None)                         #No weights

pre_trained_model.load_weights(local_weights_file)                      #Load the downloaded weights into the model

for layer in pre_trained_model.layers:                                  #Freeze all the layers from the pre_trained_model
    layer.trainable = False

# pre_trained_model.summary()                                           #Print model summary, although it's too large. Useful to know from
                                                                        #which layer start feeding the model

last_layer = pre_trained_model.get_layer('mixed7')
print('Last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

from tensorflow.keras.optimizers import RMSprop

#Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
#Add a fully connected layer with 1024 hidden units and ReLU activations
x = layers.Dense(1024, activation = 'relu')(x)
#Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)
#Add a final sigmoid layer for classification
x = layers.Dense(15, activation = 'softmax')(x)

model = Model(pre_trained_model.input, x)

model.summary()

model.compile(optimizer = RMSprop(lr = 0.001),                         ##Use small learning rate to avoid overfitting
                loss = 'categorical_crossentropy',
                metrics=['categorical_accuracy'])



class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('categorical_accuracy')>=0.98):
            print("\nReached over 98% accuracy so cancelling training!")
            self.model.stop_training = True
callbacks = myCallback()

TRAINING_DIR = r"C:\Users\janal\Downloads\DataAI project\Train_set"
training_datagen = ImageDataGenerator(
      rescale = 1./255,
      rotation_range=90,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      vertical_flip = True,
      fill_mode='nearest')

VALIDATION_DIR = r"C:\Users\janal\Downloads\DataAI project\Validation_set"
validation_datagen = ImageDataGenerator(rescale = 1./255.)

train_generator = training_datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(150,150),
	class_mode='categorical',
    batch_size=5)

validation_generator = validation_datagen.flow_from_directory(
	VALIDATION_DIR,
	target_size=(150,150),
	class_mode='categorical',
    batch_size=5)

history = model.fit_generator(train_generator,
                            epochs=250,
                            steps_per_epoch=9,
                            validation_data = validation_generator,
                            verbose = 1,
                            validation_steps=3,
                            callbacks = callbacks)

model.save(r"Trained_models\Transfer_weights_on250epochs_callback.h5")

import matplotlib.pyplot as plt
acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

fig, (plt1, plt2) = plt.subplots(1,2)
plt1.plot(epochs, acc, 'r', label='Training accuracy')
plt1.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt1.set_title('Training and validation accuracy')
plt1.legend(loc=0)

plt2.plot(epochs, loss, 'r', label='Training loss')
plt2.plot(epochs, val_loss, 'b', label='Validation loss')
plt2.set_title('Training and validation loss')
plt2.legend(loc=0)

plt.show()