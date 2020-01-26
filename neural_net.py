import keras
from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
import matplotlib.pyplot as plt


model = models.Sequential()

model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.AveragePooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(10, activation='softmax'))

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
plt.imshow(train_images[10], cmap=plt.cm.binary)
plt.show()
print(train_labels[10])

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
#train_images=train_images[0:10000]
#print(train_images.shape)

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255
#test_images=test_images[0:1000]
#print(test_images.shape)

train_labels = to_categorical(train_labels)
#train_labels=train_labels[0:10000]
test_labels = to_categorical(test_labels)
#test_labels=test_labels[0:1000]


Batch_Size = 32
Epochs = 20


datagen = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")
datagen.fit(train_images)

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

#model.fit_generator(datagen.flow(train_images, train_labels, batch_size=Batch_Size),steps_per_epoch=len(train_images) / 32, epochs=Epochs)



'''model.fit(train_images, train_labels,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1
          )'''

#plot_model(model, to_file='model.png')





