import numpy as np
import matplotlib.pyplot as plt

from keras.optimizers import *
from keras.callbacks import *
from keras import backend as K

from path import data_path, model_path

from methods.motion.model._01_data_generator import Generator
from methods.motion.model._02_models import model_classification


def lr_schedule(epoch):
    lr = 0.01
    if epoch > 100: lr /= 10
    if epoch > 200: lr /= 10
    if epoch > 300: lr /= 10
    if epoch > 400: lr /= 10
    return lr


def loss(y_true, y_pred):
    channels = y_pred.shape[-1]
    loss = K.binary_crossentropy(y_true[..., 0], y_pred[..., 0])
    loss = K.mean(loss, axis=(1, 2))

    for i in range(1, channels):
        rmse = K.square(y_true[..., i] - y_pred[..., i]) * y_true[..., 0]
        non_zero_count = K.sum(K.cast(K.not_equal(rmse, 0.0), 'float32'), axis=(1, 2))
        rmse_sum = K.sum(rmse, axis=(1, 2))
        rmse = rmse_sum / (non_zero_count + 1e+07)
        loss += rmse

    return K.mean(loss)


def train_model(model, generator, epochs=500, save_path='model.h5'):
    scheduler = LearningRateScheduler(lr_schedule)
    callbacks = [scheduler]

    model.compile(optimizer=Adam(lr=0.01), loss=loss)
    history = model.fit_generator(generator, epochs=epochs, callbacks=callbacks)

    model.save(save_path)

    plt.plot(history.history['loss'])
    plt.show()


if __name__ == '__main__':
    # Load model
    model = model_classification(channels=3, window_size=64, output_channels=1, fcn=False)

    # Set generator
    data_type = 0b000001  # DATA_CLASSIFICATION, DATA_SIGNAL_RESTORE
    y_type = 0b000000  # Y_NOISE_LENGTH, Y_BPM
    augment_type = 0b000001  # AUGMENTATION_COLOR, AUGMENTATION_NOISE
    path = data_path('train/data.h5')
    generator = Generator(path, y_threshold=0.7, batch_size=10, data_type=data_type, y_type=y_type, augment_type=augment_type)

    # Train model
    train_model(model, generator, epochs=500, save_path=model_path('detect_roi/model.h5'))
