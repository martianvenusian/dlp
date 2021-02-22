import os, pwd
import numpy as np
from numpy.core.fromnumeric import shape
from numpy.lib.histograms import histogram
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
sess.as_default()

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")


conv_base = VGG16(weights='imagenet', include_top=False,input_shape=(150,150,3))

username = pwd.getpwuid(os.getuid())[0]
base_dir = '/home/{}/Downloads/cats_and_dogs_small'.format(username)
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')
batch_size = 20

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
    )
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150,150),
        batch_size= batch_size,
        class_mode='binary',
    )

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150,150),
        batch_size= batch_size,
        class_mode='binary',
    )

# Define densely connected classifier and train
from keras import models
from keras import layers
from keras import optimizers
conv_base.summary()
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu', input_dim=4*4*512))
# model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(
    optimizer=optimizers.RMSprop(lr=1e-5),
    loss='binary_crossentropy',
    metrics=['acc']
    )

history = model.fit(
        train_generator, 
        steps_per_epoch=100,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=50,
        )

# Loss and accuracy 
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training va validation loss')
plt.legend()

plt.show()
