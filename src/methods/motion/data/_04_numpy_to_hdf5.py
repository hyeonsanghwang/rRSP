import numpy as np
import h5py

from path import data_path

if __name__ == '__main__':
    x_path = data_path('train/xs.npy')
    signal_path = data_path('train/signal.npy')
    label_path = data_path('train/label.npy')

    np_xs = np.load(x_path)
    np_signal = np.load(signal_path)
    np_label = np.load(label_path)

    result_path = data_path('train/data.h5')

    hf = h5py.File(result_path, 'w')
    hf.create_dataset('xs', data=np_xs)
    hf.create_dataset('signal', data=np_signal)
    hf.create_dataset('label', data=np_label)

    hf.close()
