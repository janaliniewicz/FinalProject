import os
from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3

local_weights_file = r'C:\Users\janal\Downloads\DataAI project\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                # Create an instance of a model with inception's model structure
                                include_top=False,  # Do not include the top layer
                                weights=None)  # No weights

pre_trained_model.load_weights(local_weights_file)  # Load the downloaded weights into the model

for layer in pre_trained_model.layers:  # Freeze all the layers from the pre_trained_model
    layer.trainable = False

# pre_trained_model.summary()                                           #Print model summary, although it's too large for this Inception model.
# Useful to know from which layer start feeding the new model

last_layer = pre_trained_model.get_layer('mixed7')  # Set the last layer that'll be left from the Inception model.
print('Last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output  # Get the oututs of that last layer, which will be fed to the new model

from tensorflow.keras.optimizers import RMSprop
import tkinter as tk
from tkinter import filedialog

# Define the new model which will go on top of the last layer from the Inception model
# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1024 hidden units and ReLU activations
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)
# Add a final softmax layer for multi-class classification
x = layers.Dense(15, activation='softmax')(x)

model = Model(pre_trained_model.input, x)

root = tk.Tk()
root.withdraw()

input('\n\n\nSelect the data weights next: \n\n\n')
pre_trained_model2 = filedialog.askopenfilename()

model.load_weights(pre_trained_model2)  # Load the weights from the previously trained model

# RUNNING THE MODEL
import numpy as np
from keras.preprocessing import image


def one_hot(index):
    array = np.zeros((1, 15))
    array[0][index] = 1
    return array


tree_classes = [1, 10, 11, 12, 13, 14, 15, 2, 3, 4, 5, 6, 7, 8,
                9]  # <-- Define the alphanumeric order of the classes, which is how tensorflow outputs the predictions
again = 'yes'

while (again == 'yes'):
    input('\n\n\nSelect test directory next: \n\n\n')
    root = tk.Tk()
    root.withdraw()
    Test_set_dir = filedialog.askdirectory()
    os.chdir(Test_set_dir)

    Incorrect_predictions = []
    Accuracy = 0
    Total_Correct_predictions = 0
    Total_predictions = 0
    i = 0  # Set counter which will tell me in which folder I am
    for Folder in os.listdir():
        Class_correct_predictions = 0
        Class_wrong_predictions = 0
        current_path = os.path.join(Test_set_dir, Folder)
        os.chdir(current_path)
        True_class = one_hot(i)  # Create array of 14 0's and one 1 where the true class is in the i'th position

        if (len(os.listdir()) != 0):  # If the folder is not empty
            print('***********************************************************************')
            for file in os.listdir():  # For each file in the current directory (Predict images)
                Total_predictions += 1
                path = os.path.abspath(file)
                img = image.load_img(path, target_size=(150, 150))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = x / 255
                images = np.vstack([x])
                classes = model.predict(images, batch_size=1)
                print(file)
                index = np.argmax(classes[0])
                probability = classes[0][index] * 100
                print('Predicted class {}, probability: {}\nTrue class: {}'.format(tree_classes[index], probability,
                                                                                   tree_classes[i]))
                Prediction_vector = one_hot(index)

                # print('True class: {}'.format(True_class))
                # print('Prediction: {}'.format(Prediction_vector))
                if ((Prediction_vector == True_class).all()):
                    Total_Correct_predictions += 1
                    Class_correct_predictions += 1
                else:
                    print('INCORRECTLY CLASSIFIED\n')
                    Class_wrong_predictions += 1
                    Incorrect_predictions.append(file)
            print('-----------------------------------------------------------------------')
            print('Correct_predictions for class {}: {}\nWrong predictions for class {}: {}'.format(tree_classes[i],
                                                                                                    Class_correct_predictions,
                                                                                                    tree_classes[i],
                                                                                                    Class_wrong_predictions))
            print('-----------------------------------------------------------------------\n\n\n')
        i += 1
    Accuracy = Total_Correct_predictions / Total_predictions
    print('Total predictions: ' + str(Total_predictions))
    print('Total correct predictions: ' + str(Total_Correct_predictions))
    print('Accuracy: ' + str(Accuracy) + '\n\n')

    for i in range(len(Incorrect_predictions)):  # Prints incorrectly predicted images
        print(Incorrect_predictions[i])
    Incorrect_predictions *= 0

    again = input("Do you want to make another prediction?: ")