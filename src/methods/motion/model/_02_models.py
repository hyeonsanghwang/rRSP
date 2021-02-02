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


def model_classification(channels=3, window_size=64, n_filters=(8, 16, 32, 64, 128), output_channels=1):
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
        fs[i] = AveragePooling3D(pool_size=(fs[i].shape[1], 1, 1), strides=(1, 1, 1))(fs[i])
        fs[i] = Lambda(lambda x: K.squeeze(x, axis=1))(fs[i])
        fs[i] = Dense(units=1, activation=activation)(fs[i])
        # fs[i] = Conv2D(filters=64, kernel_size=1, activation='relu')(fs[i])
        # fs[i] = Conv2D(filters=32, kernel_size=1, activation='relu')(fs[i])
        # fs[i] = Conv2D(filters=8, kernel_size=1, activation='relu')(fs[i])
        # fs[i] = Conv2D(filters=1, kernel_size=1, activation=activation)(fs[i])

    y = fs[0] if len(fs) == 1 else Concatenate(axis=-1)(fs)
    return Model(x, y)


def model_signal(channels=3, window_size=64, n_filters=(8, 16, 32, 64, 128)):
    x = Input((window_size, None, None, channels))

    f1 = Conv3D(filters=8, kernel_size=(3, 1, 1), padding='same', use_bias=False)(x)
    f1 = BatchNormalization()(f1)
    f1 = Activation('relu')(f1)
    f1 = MaxPooling3D(pool_size=(3, 1, 1), strides=(2, 1, 1), padding='same')(f1)

    f2 = Conv3D(filters=16, kernel_size=(3, 1, 1), padding='same', use_bias=False)(f1)
    f2 = BatchNormalization()(f2)
    f2 = Activation('relu')(f2)
    f2 = MaxPooling3D(pool_size=(3, 1, 1), strides=(2, 1, 1), padding='same')(f2)

    f3 = Conv3D(filters=32, kernel_size=(3, 1, 1), padding='same', use_bias=False)(f2)
    f3 = BatchNormalization()(f3)
    f3 = Activation('relu')(f3)
    f3 = MaxPooling3D(pool_size=(3, 1, 1), strides=(2, 1, 1), padding='same')(f3)

    f4 = Conv3D(filters=64, kernel_size=(3, 1, 1), padding='same', use_bias=False)(f3)
    f4 = BatchNormalization()(f4)
    f4 = Activation('relu')(f4)
    f4 = MaxPooling3D(pool_size=(3, 1, 1), strides=(2, 1, 1), padding='same')(f4)

    f5 = Conv3D(filters=128, kernel_size=(3, 1, 1), padding='same', use_bias=False)(f4)
    f5 = BatchNormalization()(f5)
    f5 = Activation('relu')(f5)
    f5 = MaxPooling3D(pool_size=(3, 1, 1), strides=(2, 1, 1), padding='same')(f5)

    d1 = Conv3DTranspose(filters=64, kernel_size=(3, 1, 1), padding='same', strides=(2, 1, 1))(f5)
    d1 = BatchNormalization()(d1)
    d1 = Activation('relu')(d1)

    d1 = Conv3DTranspose(filters=64, kernel_size=(3, 1, 1), padding='same', strides=(2, 1, 1))(f5)
    d1 = BatchNormalization()(d1)
    d1 = Activation('relu')(d1)

    d2 = Conv3DTranspose(filters=32, kernel_size=(3, 1, 1), padding='same', strides=(2, 1, 1))(d1)
    d2 = BatchNormalization()(d2)
    d2 = Activation('relu')(d2)

    d3 = Conv3DTranspose(filters=16, kernel_size=(3, 1, 1), padding='same', strides=(2, 1, 1))(d2)
    d3 = BatchNormalization()(d3)
    d3 = Activation('relu')(d3)

    d4 = Conv3DTranspose(filters=8, kernel_size=(3, 1, 1), padding='same', strides=(2, 1, 1))(d3)
    d4 = BatchNormalization()(d4)
    d4 = Activation('relu')(d4)

    d5 = Conv3DTranspose(filters=channels, kernel_size=(3, 1, 1), padding='same', strides=(2, 1, 1))(d4)
    return Model(x, d5)


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
