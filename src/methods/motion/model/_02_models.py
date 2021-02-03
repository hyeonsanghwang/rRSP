import numpy as np
from time import perf_counter

from keras.layers import *
from keras.models import *

from keras import backend as K


def residual_block(x, n_filter, shortcut=False, pooling=True, first_stride=1):
    f = Conv3D(filters=n_filter, kernel_size=(3, 1, 1), strides=(first_stride, 1, 1), padding='same', activation='relu')(x)
    f = Conv3D(filters=n_filter, kernel_size=(3, 1, 1), padding='same', use_bias=False)(f)
    f = BatchNormalization()(f)
    if shortcut:
        s = Conv3D(filters=n_filter, kernel_size=(1, 1, 1), strides=(first_stride, 1, 1), padding='same', activation=None)(x)
        f = Add()([s, f])
    f = Activation('relu')(f)
    if pooling:
        f = MaxPooling3D(pool_size=(3, 1, 1), strides=(2, 1, 1), padding='same')(f)
    return f


def model_classification(channels=3, window_size=64, n_filters=(8, 16, 32, 64, 128), output_channels=1, fcn=False):
    f = x = Input((window_size, None, None, channels))
    fs = [None] * output_channels
    for i, n in enumerate(n_filters):
        shortcut = False if i == 0 else True
        pooling = False if i == (len(n_filters) - 1) else True
        stride = 2 if i < 2 else 1
        if i > 2:
            for j in range(output_channels):
                fs[j] = residual_block(fs[j], n, shortcut, pooling, first_stride=stride)
        else:
            f = residual_block(f, n, shortcut, pooling, first_stride=stride)
            for j in range(output_channels):
                fs[j] = f

    for i in range(output_channels):
        activation = 'sigmoid' if i == 0 else None
        if fcn:
            fs[i] = Conv2D(filters=64, kernel_size=1, activation='relu')(fs[i])
            fs[i] = Conv2D(filters=32, kernel_size=1, activation='relu')(fs[i])
            fs[i] = Conv2D(filters=8, kernel_size=1, activation='relu')(fs[i])
            fs[i] = Conv2D(filters=1, kernel_size=1, activation=activation)(fs[i])
        else:
            fs[i] = AveragePooling3D(pool_size=(fs[i].shape[1], 1, 1), strides=(1, 1, 1))(fs[i])
            fs[i] = Lambda(lambda x: K.squeeze(x, axis=1))(fs[i])
            fs[i] = Dense(units=1, activation=activation)(fs[i])

    y = fs[0] if len(fs) == 1 else Concatenate(axis=-1)(fs)
    return Model(x, y)


def model_signal(channels=3, window_size=64, n_filters=(8, 16, 32, 64, 128)):
    x = Input((window_size, None, None, channels))

    f = x
    for n_filter in n_filters:
        f = Conv3D(filters=n_filter, kernel_size=(3, 1, 1), padding='same', use_bias=False)(f)
        f = BatchNormalization()(f)
        f = Activation('relu')(f)
        f = MaxPooling3D(pool_size=(3, 1, 1), strides=(2, 1, 1), padding='same')(f)

    d = f
    for n_filter in n_filters[:-1][::-1]:
        d = Conv3DTranspose(filters=n_filter, kernel_size=(3, 1, 1), padding='same', strides=(2, 1, 1))(d)
        d = BatchNormalization()(d)
        d = Activation('relu')(d)

    s = Conv3DTranspose(filters=channels, kernel_size=(3, 1, 1), padding='same', strides=(2, 1, 1))(d)
    return Model(x, s)


def speed_test(model, data_size=8, n_iter=100):
    shape = (1, model.input.shape[1], 480 // data_size, 640 // data_size, model.input.shape[-1])

    # Skip first predict time
    x = np.random.rand(*shape)
    model.predict(x)
    x = np.random.rand(*shape)
    model.predict(x)

    # Time measurement
    print("> Model speed test")
    start = perf_counter()
    for i in range(n_iter):
        x = np.random.rand(*shape)
        model.predict(x)
    end = perf_counter()
    perf_time = (end - start) / n_iter
    print("Performance time : ", perf_time, "/ FPS : ", 1.0 / perf_time)


if __name__ == '__main__':
    model = model_classification(channels=3, window_size=64, n_filters=[16, 32, 64, 128, 256], output_channels=1)
    # model = model_signal(channels=3, window_size=64, n_filters=[8, 16, 32, 64, 128])
    model.summary()
    speed_test(model, data_size=16)
