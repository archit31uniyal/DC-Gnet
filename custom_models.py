import keras

from keras import losses
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Input, MaxPooling2D, AveragePooling2D, average
from keras.layers import concatenate, Conv2D, Conv2DTranspose, Dropout
from keras.models import Model
from keras.optimizers import Adadelta

from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, UpSampling2D
from keras.layers import Convolution2D, ZeroPadding2D, Embedding, LSTM, merge, Lambda, Deconvolution2D, Cropping2D

from keras.layers import ELU, ReLU
act = ReLU


def get_unet(do=0, activation=act):
    inputs = Input((None, None, 3))
    conv1 = Dropout(do)(activation()(Conv2D(32, (4, 4), padding='same')(inputs)))
    conv1 = Dropout(do)(activation()(Conv2D(32, (4, 4), padding='same')(conv1)))
    pool1 = MaxPooling2D(pool_size=(4, 4))(conv1)

    conv2 = Dropout(do)(activation()(Conv2D(64, (4, 4), padding='same')(pool1)))
    conv2 = Dropout(do)(activation()(Conv2D(64, (4, 4), padding='same')(conv2)))
    pool2 = MaxPooling2D(pool_size=(4, 4))(conv2)

    conv3 = Dropout(do)(activation()(Conv2D(128, (4, 4), padding='same')(pool2)))
    conv3 = Dropout(do)(activation()(Conv2D(128, (4, 4), padding='same')(conv3)))
    pool3 = MaxPooling2D(pool_size=(4, 4))(conv3)

    conv4 = Dropout(do)(activation()(Conv2D(128, (4, 4), padding='same')(pool3)))
    conv4 = Dropout(do)(activation()(Conv2D(128, (4, 4), padding='same')(conv4)))
    pool4 = MaxPooling2D(pool_size=(4, 4))(conv4)

    conv5 = Dropout(do)(activation()(Conv2D(128, (4, 4), padding='same')(pool4)))
    conv5 = Dropout(do)(activation()(Conv2D(128, (4, 4), padding='same')(conv5)))

    up6 = concatenate([UpSampling2D(size=(4, 4))(conv5), conv4], axis=3)
    conv6 = Dropout(do)(activation()(Conv2D(128, (3, 3), padding='same')(up6)))
    conv6 = Dropout(do)(activation()(Conv2D(128, (3, 3), padding='same')(conv6)))

    up7 = concatenate([UpSampling2D(size=(4, 4))(conv6), conv3], axis=3)
    conv7 = Dropout(do)(activation()(Conv2D(64, (3, 3), padding='same')(up7)))
    conv7 = Dropout(do)(activation()(Conv2D(64, (3, 3), padding='same')(conv7)))

    up8 = concatenate([UpSampling2D(size=(4, 4))(conv7), conv2], axis=3)
    conv8 = Dropout(do)(activation()(Conv2D(64, (3, 3), padding='same')(up8)))
    conv8 = Dropout(do)(activation()(Conv2D(64, (3, 3), padding='same')(conv8)))

    up9 = concatenate([UpSampling2D(size=(4, 4))(conv8), conv1], axis=3)
    conv9 = Dropout(do)(activation()(Conv2D(32, (3, 3), padding='same')(up9)))
    conv9 = Dropout(do)(activation()(Conv2D(32, (3, 3), padding='same')(conv9)))

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    return model


def DeepModel(size_set=256):
    img_input = Input(shape=(size_set, size_set, 3))

    scale_img_2 = AveragePooling2D(pool_size=(2, 2))(img_input)
    scale_img_3 = AveragePooling2D(pool_size=(2, 2))(scale_img_2)
    scale_img_4 = AveragePooling2D(pool_size=(2, 2))(scale_img_3)

    conv1 = Conv2D(32, (3, 3), padding='same', activation='relu', name='block1_conv1')(img_input)
    conv1 = Conv2D(32, (3, 3), padding='same', activation='relu', name='block1_conv2')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    input2 = Conv2D(64, (3, 3), padding='same', activation='relu', name='block2_input1')(scale_img_2)
    input2 = concatenate([input2, pool1], axis=3)
    conv2 = Conv2D(64, (3, 3), padding='same', activation='relu', name='block2_conv1')(input2)
    conv2 = Conv2D(64, (3, 3), padding='same', activation='relu', name='block2_conv2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    input3 = Conv2D(128, (3, 3), padding='same', activation='relu', name='block3_input1')(scale_img_3)
    input3 = concatenate([input3, pool2], axis=3)
    conv3 = Conv2D(128, (3, 3), padding='same', activation='relu', name='block3_conv1')(input3)
    conv3 = Conv2D(128, (3, 3), padding='same', activation='relu', name='block3_conv2')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    input4 = Conv2D(256, (3, 3), padding='same', activation='relu', name='block4_input1')(scale_img_4)
    input4 = concatenate([input4, pool3], axis=3)
    conv4 = Conv2D(256, (3, 3), padding='same', activation='relu', name='block4_conv1')(input4)
    conv4 = Conv2D(256, (3, 3), padding='same', activation='relu', name='block4_conv2')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_conv1')(pool4)
    conv5 = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_conv2')(conv5)

    up6 = concatenate(
        [Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', name='block6_dconv')(conv5), conv4],
        axis=3)
    conv6 = Conv2D(256, (3, 3), padding='same', activation='relu', name='block6_conv1')(up6)
    conv6 = Conv2D(256, (3, 3), padding='same', activation='relu', name='block6_conv2')(conv6)

    up7 = concatenate(
        [Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', name='block7_dconv')(conv6), conv3],
        axis=3)
    conv7 = Conv2D(128, (3, 3), padding='same', activation='relu', name='block7_conv1')(up7)
    conv7 = Conv2D(128, (3, 3), padding='same', activation='relu', name='block7_conv2')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', name='block8_dconv')(conv7), conv2],
                      axis=3)
    conv8 = Conv2D(64, (3, 3), padding='same', activation='relu', name='block8_conv1')(up8)
    conv8 = Conv2D(64, (3, 3), padding='same', activation='relu', name='block8_conv2')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', name='block9_dconv')(conv8), conv1],
                      axis=3)
    conv9 = Conv2D(32, (3, 3), padding='same', activation='relu', name='block9_conv1')(up9)
    conv9 = Conv2D(32, (3, 3), padding='same', activation='relu', name='block9_conv2')(conv9)

    side6 = UpSampling2D(size=(8, 8))(conv6)
    side7 = UpSampling2D(size=(4, 4))(conv7)
    side8 = UpSampling2D(size=(2, 2))(conv8)
    out6 = Conv2D(1, (1, 1), activation='sigmoid', name='side_63')(side6)
    out7 = Conv2D(1, (1, 1), activation='sigmoid', name='side_73')(side7)
    out8 = Conv2D(1, (1, 1), activation='sigmoid', name='side_83')(side8)
    out9 = Conv2D(1, (1, 1), activation='sigmoid', name='side_93')(conv9)

    out10 = average([out6, out7, out8, out9])

    return Model(inputs=[img_input], outputs=[out10])


def get_unet1(do=0, activation=act):
    inputs = Input((None, None, 3))
    conv1 = Dropout(do)(activation()(Conv2D(32, (4, 4), padding='same')(inputs)))
    conv1 = Dropout(do)(activation()(Conv2D(32, (4, 4), padding='same')(conv1)))
    pool1 = MaxPooling2D(pool_size=(4, 4))(conv1)

    conv2 = Dropout(do)(activation()(Conv2D(64, (4, 4), padding='same')(pool1)))
    conv2 = Dropout(do)(activation()(Conv2D(64, (4, 4), padding='same')(conv2)))
    pool2 = MaxPooling2D(pool_size=(4, 4))(conv2)

    conv3 = Dropout(do)(activation()(Conv2D(64, (4, 4), padding='same')(pool2)))
    conv3 = Dropout(do)(activation()(Conv2D(64, (4, 4), padding='same')(conv3)))
    pool3 = MaxPooling2D(pool_size=(4, 4))(conv3)

    conv4 = Dropout(do)(activation()(Conv2D(128, (4, 4), padding='same')(pool3)))
    conv4 = Dropout(do)(activation()(Conv2D(128, (4, 4), padding='same')(conv4)))
    pool4 = MaxPooling2D(pool_size=(4, 4))(conv4)

    conv5 = Dropout(do)(activation()(Conv2D(256, (4, 4), padding='same')(pool4)))
    conv5 = Dropout(do)(activation()(Conv2D(256, (4, 4), padding='same')(conv5)))

    up6 = concatenate([UpSampling2D(size=(4, 4))(conv5), conv4], axis=3)
    conv6 = Dropout(do)(activation()(Conv2D(128, (4, 4), padding='same')(up6)))
    conv6 = Dropout(do)(activation()(Conv2D(128, (4, 4), padding='same')(conv6)))

    up7 = concatenate([UpSampling2D(size=(4, 4))(conv6), conv3], axis=3)
    conv7 = Dropout(do)(activation()(Conv2D(64, (4, 4), padding='same')(up7)))
    conv7 = Dropout(do)(activation()(Conv2D(64, (4, 4), padding='same')(conv7)))

    up8 = concatenate([UpSampling2D(size=(4, 4))(conv7), conv2], axis=3)
    conv8 = Dropout(do)(activation()(Conv2D(64, (4, 4), padding='same')(up8)))
    conv8 = Dropout(do)(activation()(Conv2D(64, (4, 4), padding='same')(conv8)))

    up9 = concatenate([UpSampling2D(size=(4, 4))(conv8), conv1], axis=3)
    conv9 = Dropout(do)(activation()(Conv2D(32, (4, 4), padding='same')(up9)))
    conv9 = Dropout(do)(activation()(Conv2D(32, (4, 4), padding='same')(conv9)))

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    return model

