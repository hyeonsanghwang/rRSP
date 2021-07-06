import numpy as np
import cv2
import h5py

from keras.utils import Sequence

from project_utils.path import data_path


class Generator(Sequence):
    DATA_CLASSIFICATION = 0b000001
    DATA_SIGNAL_RESTORE = 0b000010

    Y_NOISE_LENGTH = 0b000001
    Y_BPM = 0b000010

    AUGMENTATION_COLOR = 0b000001
    AUGMENTATION_NOISE = 0b000010

    def __init__(self, path, y_threshold=0.6, batch_size=32, data_type=DATA_CLASSIFICATION, y_type=0, augment_type=0):
        hf = h5py.File(path, 'r')
        self.xs = hf.get('xs')
        self.ys = hf.get('label')
        self.y_threshold = y_threshold

        self.batch_size = batch_size

        self.data_type = data_type
        self.y_type = y_type
        self.augment_type = augment_type

        self.index = None

        self.on_epoch_end()

    def on_epoch_end(self):
        self.index = np.arange(self.xs.shape[0])
        np.random.shuffle(self.index)

    def __len__(self):
        return int(np.floor(self.xs.shape[0] / self.batch_size))

    def __getitem__(self, batch_index):
        idx = self.index[batch_index * self.batch_size: (batch_index + 1) * self.batch_size]
        idx = np.sort(idx)
        x = self.xs[idx].astype(np.float32) / 255.0
        y = (self.xs[idx].astype(np.float32) / 255.0) if self.data_type & self.DATA_SIGNAL_RESTORE else (self.ys[idx] >= self.y_threshold).astype(np.float32)
        bpm = self.bpms[idx] if self.y_type & self.Y_BPM else None

        x, y = self.augmentation(x, y, bpm)
        return x, y

    def augmentation(self, xs, ys, bpms):
        if self.data_type & self.DATA_SIGNAL_RESTORE:
            return self.signal_restore_data(xs, ys)

        elif self.data_type & self.DATA_CLASSIFICATION: # Augmentation

            # Color augmentation
            if self.augment_type & self.AUGMENTATION_COLOR:
                xs = self.augmentation_color(xs)

            # Noise augmentation
            noise_label = None
            if self.augment_type & self.AUGMENTATION_NOISE:
                xs, noise_label = self.augmentation_noise(xs, ret_noise_label=True)

            # Append additional labels
            if self.y_type & self.Y_NOISE_LENGTH:   # append noise length
                ys = self.label_noise_length(ys, noise_label)
            if self.y_type & self.Y_BPM:            # append bpm
                ys = self.label_bpm(ys, bpms)

            # return classification label
            return xs, ys
        else:
            print('Either Y_SIGNAL or Y_CLASSIFICATION must be included.')
            return None, None

    def augmentation_color(self, xs):
        for i in range(xs.shape[0]):
            if np.random.rand() > 0.5:
                xs[i, :, :, :, 0] = 1 - xs[i, :, :, :, 0]
            if np.random.rand() > 0.5:
                xs[i, :, :, :, 1] = 1 - xs[i, :, :, :, 1]
            if np.random.rand() > 0.5:
                xs[i, :, :, :, 2] = 1 - xs[i, :, :, :, 2]

            if np.random.rand() > 0.5:
                xs[i, :, :, :, 0] = xs[i, :, :, :, 0] * (np.random.rand() * 2)
            if np.random.rand() > 0.5:
                xs[i, :, :, :, 1] = xs[i, :, :, :, 1] * (np.random.rand() * 2)
            if np.random.rand() > 0.5:
                xs[i, :, :, :, 2] = xs[i, :, :, :, 2] * (np.random.rand() * 2)
        xs[xs > 1.0] = 1.0
        xs[xs < 0] = 0
        return xs

    def augmentation_noise(self, xs, ret_noise_label=False):
        signal_length = xs.shape[1]
        noise_label = []
        for x in xs:
            noise_length = int(np.random.uniform(0, signal_length // 4))
            start_idx = int(np.random.uniform(0, signal_length - noise_length))
            end_idx = start_idx + noise_length

            noise = np.random.normal(np.random.uniform(0, 1),
                                     np.random.uniform(0, 1),
                                     size=(noise_length,
                                           x.shape[1],
                                           x.shape[2],
                                           x.shape[3]))
            x[start_idx: end_idx, ...] = noise

            def flip_signal(x):
                flipped = 1 - x
                scale = flipped.max() - flipped.min()
                start_val = np.random.uniform(0, 1 - scale)
                flipped = start_val + flipped - flipped.min()
                flipped[flipped > 1] = 1
                flipped[flipped < 0] = 0
                return flipped

            is_flip = bool(np.random.randint(2))
            if is_flip:
                if start_idx > (signal_length - end_idx):
                    if end_idx < signal_length:
                        x[end_idx:] = flip_signal(x[end_idx:])
                else:
                    if start_idx > 0:
                        x[:start_idx] = flip_signal(x[:start_idx])
            noise_label.append([noise_length / signal_length, start_idx / signal_length, end_idx / signal_length])

        xs[xs > 1.0] = 1.0
        xs[xs < 0] = 0

        if ret_noise_label:
            return xs, noise_label
        else:
            return xs

    def signal_restore_data(self, xs, ys):
        xs = self.augmentation_noise(xs)
        return xs, ys

    def label_noise_length(self, ys, noise_label):
        ys_noise_len = np.ones_like(ys[..., 0:1])
        ys_noise_start = np.ones_like(ys[..., 0:1])
        ys_noise_end = np.ones_like(ys[..., 0:1])

        if noise_label is not None:
            noise_label = np.array(noise_label)
            ys_noise_len *= noise_label[..., 0].reshape((noise_label.shape[0], 1, 1, 1))
            ys_noise_start *= noise_label[..., 1].reshape((noise_label.shape[0], 1, 1, 1))
            ys_noise_end *= noise_label[..., 2].reshape((noise_label.shape[0], 1, 1, 1))

        ys = np.append(ys, ys_noise_len, axis=-1)
        ys = np.append(ys, ys_noise_start, axis=-1)
        ys = np.append(ys, ys_noise_end, axis=-1)

        return ys

    def label_bpm(self, ys, bpms):
        return np.append(ys, bpms, axis=-1)


def generator_test(generator):
    for i in range(generator.__len__()):
        x_batch, y_batch = generator.__getitem__(i)
        for xs, ys in zip(x_batch, y_batch):
            if generator.data_type & Generator.DATA_SIGNAL_RESTORE:
                for x, y, in zip(xs, ys):
                    cv2.imshow('Frame', cv2.resize(x, dsize=(640, 480), interpolation=cv2.INTER_AREA))
                    cv2.imshow('Label', cv2.resize(y, dsize=(640, 480), interpolation=cv2.INTER_AREA))
                    key = cv2.waitKey(30)
                    if key == 27:
                        return
                    elif key == 13:
                        break
            elif generator.data_type & Generator.DATA_CLASSIFICATION:
                cv2.imshow('Score', cv2.resize(ys[..., 0], dsize=(640, 480), interpolation=cv2.INTER_AREA))
                idx = 1
                if generator.y_type & Generator.Y_NOISE_LENGTH:
                    cv2.imshow('Noise length', cv2.resize(ys[..., idx], dsize=(640, 480), interpolation=cv2.INTER_AREA))
                    cv2.imshow('Noise start index', cv2.resize(ys[..., idx+1], dsize=(640, 480), interpolation=cv2.INTER_AREA))
                    cv2.imshow('Noise end index', cv2.resize(ys[..., idx+2], dsize=(640, 480), interpolation=cv2.INTER_AREA))
                    idx += 3
                if generator.y_type & Generator.Y_BPM:
                    cv2.imshow('BPM', cv2.resize(ys[..., idx], dsize=(640, 480), interpolation=cv2.INTER_AREA))
                    idx += 1

                for x in xs:
                    cv2.imshow('Frame', cv2.resize(x, dsize=(640, 480), interpolation=cv2.INTER_AREA))
                    key = cv2.waitKey(30)
                    if key == 27:
                        return
                    elif key == 13:
                        break


if __name__ == '__main__':
    ## ******************* Set generator type ******************* ##
    data_type = 0b000001    # DATA_CLASSIFICATION, DATA_SIGNAL_RESTORE
    y_type = 0b000001       # Y_NOISE_LENGTH, Y_BPM
    augment_type = 0b000011 # AUGMENTATION_COLOR, AUGMENTATION_NOISE
    path = data_path('train/data.h5')

    generator = Generator(path, y_threshold=0.7, batch_size=10, data_type=data_type, y_type=y_type, augment_type=augment_type)

    ## ********************* Test generator ********************* ##
    generator_test(generator)
    cv2.destroyAllWindows()
