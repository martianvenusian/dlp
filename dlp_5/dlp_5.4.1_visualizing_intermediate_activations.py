from keras.models import load_model
model = load_model('cats_and_dogs_small_2.h5')
model.summary()

import os, pwd
username = pwd.getpwuid(os.getuid())[0]
base_dir = '/home/{}/Downloads/cats_and_dogs_small'.format(username)
test_dir = os.path.join(base_dir, 'test')
img_path = '{}/cats/1700.jpg'.format(test_dir)
print(img_path)
from keras.preprocessing import image
import numpy as np
img = image.load_img(img_path, target_size=(150,150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.
print(img_tensor.shape)

# display the picture
import matplotlib.pyplot as plt
plt.imshow(img_tensor[0])
plt.show()

# extract feature maps
from keras import models
layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(img_tensor)

# activation of the first convolution layer for the input image
first_layer_activation = activations[0]
print(first_layer_activation.shape)
# plt.matshow(first_layer_activation[0,:,:,4], cmap='viridis')
plt.matshow(first_layer_activation[0,:,:,7], cmap='viridis')
plt.show()

# Names of the layers, so you can have them as part of your plot
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)
images_per_row = 16
# Displays the feature maps
for layer_name, layer_activation in zip(layer_names, activations):
    # Number of features in the feature map
    n_features = layer_activation.shape[-1]
    # the feature map has shape (I, size, size, n_features)
    size = layer_activation.shape[1]
    # Tiles the activation channels in this matrix
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    # Tiles each filter into a big horizontal grid
    for col in range(n_cols):
        for row in range(images_per_row):
            # Post-processes the feature to make it visually palatable
            channel_image = layer_activation[0, :, :, col * images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            # Displays the grid
            display_grid[col * size : (col + 1) * size,
            row * size : (row + 1) * size] = channel_image
    
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
    scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()

