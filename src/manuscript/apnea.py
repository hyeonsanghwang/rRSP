import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    font_size = 14

    plt.figure(figsize=(13, 6))
    for i in range(6):
        name = str(i) + '.npy'
        data = np.load(name)
        data = (data - data.min()) / (data.max() - data.min())
        data = data * 2 - 1
        plt.subplot(2, 3, i+1)

        plt.ylim(-1.1, 1.1)
        plt.xlabel('Frame #', fontsize=font_size)
        plt.ylabel('Normalized amplitude', fontsize=font_size)
        plt.plot(data, linewidth=4)

    plt.tight_layout()
    plt.show()
