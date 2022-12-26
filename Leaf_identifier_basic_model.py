import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
    batch_size=8)

validation_generator = validation_datagen.flow_from_directory(
	VALIDATION_DIR,
	target_size=(150,150),
	class_mode='categorical',
    batch_size=5)

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    #512 FC layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    # Output layer
    tf.keras.layers.Dense(15, activation='softmax')
])


model.summary()

tf.keras.optimizers.RMSprop(learning_rate = 0.001)
model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit(train_generator, epochs=250, steps_per_epoch=7, validation_data = validation_generator, verbose = 1, validation_steps=3)

model.save(r"Trained_models\BasicModel_250epochs.h5")


import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

fig, (plt1, plt2) = plt.subplots(1,2)
# fig.suptitle('Accuracy and loss plots')
plt1.plot(epochs, acc, 'r', label='Training accuracy')
plt1.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt1.set_title('Training and validation accuracy')
plt1.legend(loc=0)

# plot2 = plt.figure(2)
plt2.plot(epochs, loss, 'r', label='Training loss')
plt2.plot(epochs, val_loss, 'b', label='Validation loss')
plt2.set_title('Training and validation loss')
plt2.legend(loc=0)

plt.show()