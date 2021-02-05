import sys
from PyQt5.QtWidgets import QApplication

from methods.motion.test.window import MainWindow
from utils.visualization.signal import signal_to_frame

import numpy as np
import cv2
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()




from resp_utils.resp_signal import SignalStream
from utils.camera.video_stream import VideoStream
from utils.visualization.signal import show_sin_signals
from path import model_path, data_path

from methods.motion.data._00_data_preprocess import ColorDomain

from methods.motion.test.clustering import ClusteringTester


def set_video_src(target=0):
    """
    :param target: (0 - webcam / 1 - npy / 2 - mp4)
    :return: video_src
    """
    if target == 0:
        return 0
    elif target == 1:
        return data_path('npy/1_1_frames.npy')
    else:
        return data_path('video.mp4')


def set_signal_src(target=0):
    """
    :param target: (0 - sensor / 1 - npy)
    :return: signal_src
    """
    if target == 0:
        return None
    else:
        return data_path('npy/1_1_signal.npy')




if __name__ == '__mai1n__':
    # Set process parameters
    FPS = 60
    PROCESS_FPS = 5
    WINDOW_SIZE = 64
    RESIZE_RATIO = 24
    COLOR_DOMAIN = ColorDomain.RGB
    MODEL_PATH = model_path('detect_roi/model.h5')
    ROI_THRESHOLD = 0.5
    USE_SENSOR = True

    # Set video stream
    video_src = set_video_src(0)
    video_stream = VideoStream(video_src, FPS, WINDOW_SIZE)

    # Set signal stream
    if USE_SENSOR:
        signal_src = None
        signal_stream = SignalStream(signal_src, window_period=WINDOW_SIZE / PROCESS_FPS, show=False)

    # Set tester
    tester = OpticalFlowTester(FPS, PROCESS_FPS, WINDOW_SIZE, RESIZE_RATIO, MODEL_PATH, COLOR_DOMAIN)
    # tester = ClusteringTester(FPS, PROCESS_FPS, WINDOW_SIZE, RESIZE_RATIO, MODEL_PATH, COLOR_DOMAIN)

    # Modifiable parameters in processing time
    show_reference_signal = False
    show_selected_pixel = False
    show_mode = 1

    # Process
    while True:
        # Show reference signal
        if show_reference_signal:
            show_bpms(FPS, duration=WINDOW_SIZE / PROCESS_FPS, bpms=[10, 15, 20, 25, 30])

        # Get frame data
        ret, frame = video_stream.read()
        if not ret:
            break
        # frame = cv2.resize(frame, (640, 360))

        # Get signal data
        signal = signal_stream.get_signal() if USE_SENSOR else None

        # Main process
        tester.process(frame=frame,
                       signal=signal,
                       fps=video_stream.get_window_fps(),
                       roi_threshold=ROI_THRESHOLD,
                       show_mode=show_mode,
                       show_resized_frame=False,
                       show_model_predicts=False,
                       show_roi=False,
                       show_method_result=True,
                       show_selected_pixel=show_selected_pixel)

        # Waiting and handling events
        key=0
        # key = cv2.waitKey(video_stream.delay())
        # key = cv2.waitKey(0)
        if key == 27:
            try:
                tester.file.close()
            except:
                print('File close error')
            break
        elif key == 13:
            cv2.waitKey(0)
        elif key == ord('r'):
            show_reference_signal = not show_reference_signal
        elif ord('0') + tester.SHOW_MODES[0] <= key <= ord('0') + tester.SHOW_MODES[-1]:
            show_mode = key - ord('0')
        elif key == ord('s'):
            show_selected_pixel = not show_selected_pixel


    # Release resource
    cv2.destroyAllWindows()
    video_stream.release()
    tester.video_writer.release()
    if USE_SENSOR:
        signal_stream.close()
