import sys

from keras import applications
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Input, InputLayer, Conv2D, Activation, LeakyReLU, Concatenate, Lambda
from layers import BilinearUpSampling2D
from loss import depth_loss_function
import tensorflow as tf


def split_input(input):
    has_locations = K.equal(K.shape(input)[-1], 5)
    def split_locations():
        return tuple(tf.split(input, [3, 2], -1))
    image, aux_output = tf.cond(has_locations, split_locations, lambda: (input, tf.constant([])))
    image.set_shape((None, None, None, 3))
    return [image, aux_output, has_locations]


def concat_output(args):
    net, locations, has_locations = args
    def append_aux_output():
        zoomed = tf.image.resize_bilinear(net, tf.shape(locations)[1:3])
        return tf.concat([zoomed, locations], -1)
    return tf.cond(tf.cast(has_locations, dtype=tf.bool), append_aux_output, lambda: net)


def create_model(existing='', is_twohundred=False, is_halffeatures=True, weights='imagenet'):

    if len(existing) == 0:
        print('Loading base model (DenseNet)..')

        input = Input(shape=(None, None, None))
        image, aux_output, has_locations = Lambda(split_input)(input)

        # Encoder Layers
        if is_twohundred:
            base_model = applications.DenseNet201(input_tensor=image, weights=weights, include_top=False)
        else:
            base_model = applications.DenseNet169(input_tensor=image, weights=weights, include_top=False)


        # Starting point for decoder
        base_model_output_shape = base_model.layers[-1].output.shape
        print('Base model loaded [output: %s].' % (base_model_output_shape,))

        # Layer freezing?
        for layer in base_model.layers: layer.trainable = True

        # Starting number of decoder filters
        if is_halffeatures:
            decode_filters = int(int(base_model_output_shape[-1])/2)
        else:
            decode_filters = int(base_model_output_shape[-1])

        # Define upsampling layer
        def upproject(tensor, filters, name, concat_with):
            up_i = BilinearUpSampling2D((2, 2), name=name+'_upsampling2d')(tensor)
            up_i = Concatenate(name=name+'_concat')([up_i, base_model.get_layer(concat_with).output]) # Skip connection
            up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convA')(up_i)
            up_i = LeakyReLU(alpha=0.2)(up_i)
            up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convB')(up_i)
            up_i = LeakyReLU(alpha=0.2)(up_i)
            return up_i

        # Decoder Layers
        decoder = Conv2D(filters=decode_filters, kernel_size=1, padding='same', input_shape=base_model_output_shape, name='conv2')(base_model.output)

        decoder = upproject(decoder, int(decode_filters/2), 'up1', concat_with='pool3_pool')
        decoder = upproject(decoder, int(decode_filters/4), 'up2', concat_with='pool2_pool')
        decoder = upproject(decoder, int(decode_filters/8), 'up3', concat_with='pool1')
        decoder = upproject(decoder, int(decode_filters/16), 'up4', concat_with='conv1/relu')
        if False: decoder = upproject(decoder, int(decode_filters/32), 'up5', concat_with='input_1')

        # Extract depths (final layer)
        conv3 = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', name='conv3')(decoder)

        output = Lambda(concat_output)([conv3, aux_output, has_locations])

        output._keras_shape = output.get_shape().as_list()

        # Create the model
        model = Model(inputs=input, outputs=output)
    else:
        # Load model from file
        if not existing.endswith('.h5'):
            sys.exit('Please provide a correct model file when using [existing] argument.')
        custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': depth_loss_function}
        model = load_model(existing, custom_objects=custom_objects)
        print('\nExisting model loaded.\n')

    print('Model created.')

    return model
