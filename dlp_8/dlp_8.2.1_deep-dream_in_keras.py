from keras.applications import inception_v3
from keras import backend as K

# You won’t be training the model, so
# this command disables all training-
# specific operations.
K.set_learning_phase(0)

# Builds the Inception V3 network,
# without its convolutional base.
# The model will be loaded with
# pretrained ImageNet weights.
model= inception_v3.InceptionV3(weights='imagenet', include_top=False)

# Dictionary mapping layer names to a coefficient quantifying
# how much the layer’s activation contributes to the loss
# you’ll seek to maximize. Note that the layer names are
# hardcoded in the built-in Inception V3 application. You can
# list all layer names using model.summary().
layer_contributions = {
'mixed2': 0.2,
'mixed3': 3.,
'mixed4': 2.,
'mixed5': 1.5,
}


# Creates a dictionary that maps
# layer names to layer instances
layer_dict = dict([(layer.name, layer) for layer in model.layers])

# You’ll define the loss by adding
# layer contributions to this
# scalar variable.


loss = K.variable(0.)
for layer_name in layer_contributions:
    coeff = layer_contributions[layer_name]
    
    # Retrieves the layer’s output
    activation = layer_dict[layer_name].output
    scaling = K.prod(K.cast(K.shape(activation), 'float32'))
    print(scaling)
    # Adds the L2 norm of the features of a layer
    # to the loss. You avoid border artifacts by
    # only involving nonborder pixels in the loss.
    loss += coeff * K.sum(K.square(activation[:, 2: -2, 2: -2, :]))/ scaling

