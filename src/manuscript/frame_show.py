import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

from scipy.signal import find_peaks

from utils.visualization.signal import show_signal, signal_to_frame
from utils.processing.normalize import zero_centered_normalize


path = 'D:/respiration/npy/gui2/4_frame_*'
paths = glob.glob(path)

for path in paths:
    print(path)
    frames = np.load(path)
    for i, frame in enumerate(frames):
        print(i)

        cv2.imshow('frame', frame)
        key = cv2.waitKey(0)
        if key == 27:
            break
        if key == ord('s'):
            cv2.imwrite('C:/Users/hyeon/Desktop/'+str(i)+'.png', frame)
        if key == 13:
            break
    if key == 27:
        break

cv2.destroyAllWindows()
