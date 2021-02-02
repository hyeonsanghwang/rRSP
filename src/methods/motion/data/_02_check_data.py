import numpy as np
import cv2

from path import data_path
from utils.camera.video_stream import VideoStream
from utils.visualization.signal import show_signal, show_sin_signals
from utils.visualization.common import draw_fps
from utils.processing.normalize import zero_centered_normalize
from resp_utils.resp_signal import SignalStream

from methods.motion.data._00_data_preprocess import ColorDomain, frame_process
from methods.motion.data._00_similarity_method import *


px, py = 0, 0
def callback(e, x, y, f, p):
    if e == cv2.EVENT_LBUTTONUP:
        global px, py
        px = x // DATA_SIZE
        py = y // DATA_SIZE
        show_selected_signal()


def show_selected_signal():
    global xs, px, py
    selected = xs[:, py, px, ...]
    if len(selected.shape) == 1:
        show_signal('Channel', selected, 500)
    else:
        show_signal('Channel 1', selected[:, 0], 500)
        show_signal('Channel 2', selected[:, 1], 500)
        show_signal('Channel 3', selected[:, 2], 500)


if __name__ == '__main__':
    # Set parameters
    FPS = 10
    WINDOW_SIZE = 64
    DATA_SIZE = 8
    COLOR = ColorDomain.RGB

    # Set capture device source
    # video_src = 0
    # signal_src = None
    video_src = data_path('npy/3_1_frames.npy')
    signal_src = data_path('npy/3_1_signal.npy')

    # Set capture instance
    video_stream = VideoStream(src=video_src, fps=FPS, time_window=WINDOW_SIZE)
    signal_stream = SignalStream(src=signal_src, window_period=WINDOW_SIZE/FPS)

    # Set mouse callback
    RESIZED_NAME = 'Resized frame'
    LABEL_NAME = 'Label'
    cv2.namedWindow(RESIZED_NAME)
    cv2.namedWindow(LABEL_NAME)
    cv2.setMouseCallback(RESIZED_NAME, callback)
    cv2.setMouseCallback(LABEL_NAME, callback)

    # Process
    frames = []
    signal = []
    while True:
        show_sin_signals(fps=FPS, window_size=WINDOW_SIZE)

        # Get frame and signal data
        ret, frame = video_stream.read()
        signal_value = signal_stream.get_signal_value()
        if not ret:
            break

        # Set data
        frames.append(frame_process(frame, COLOR, DATA_SIZE))
        signal.append(signal_value)
        if len(frames) > WINDOW_SIZE:
            del frames[0]
            del signal[0]

        # Data to numpy array
        xs = np.array(frames)
        ys = np.array(signal)

        # Main process
        if len(frames) == WINDOW_SIZE:
            # Calculate similarity
            xs = zero_centered_normalize(xs, axis=0)
            ys = zero_centered_normalize(ys)

            # Calculate score for labeling
            # score = old_label_method(xs, ys)   # Old method
            score = get_pearson_correlation_score(xs, ys)
            thres = 0.7
            label = score.copy()
            label[label >= thres] = 1
            label[label < thres] = 0

            cv2.imshow('Score', cv2.resize(score, dsize=None, fx=DATA_SIZE, fy=DATA_SIZE, interpolation=cv2.INTER_AREA))
            cv2.imshow(LABEL_NAME, cv2.resize(label, dsize=None, fx=DATA_SIZE, fy=DATA_SIZE, interpolation=cv2.INTER_AREA))

        # Visualize results
        frame = draw_fps(frame, fps=round(video_stream.get_fps()))
        cv2.imshow('Frame', frame)
        show_signal('Signal', signal, 500)

        cv2.imshow(RESIZED_NAME,
                   cv2.resize(frames[-1],
                              dsize=None,
                              fx=DATA_SIZE,
                              fy=DATA_SIZE,
                              interpolation=cv2.INTER_AREA))
        show_selected_signal()

        # Wait and key event
        key = cv2.waitKey(video_stream.delay())
        if key == 27:
            break
        elif key == ord('s'):
            cv2.waitKey(0)

    # Release resources
    cv2.destroyAllWindows()
    video_stream.release()
    signal_stream.close()

