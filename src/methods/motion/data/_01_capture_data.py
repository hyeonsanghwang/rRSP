import cv2
import numpy as np

from path import data_path

from resp_utils.resp_signal import SignalStream
from utils.camera.video_stream import VideoStream
from utils.visualization.common import draw_fps
from utils.visualization.signal import show_signal, show_sin_signals


if __name__ == '__main__':
    # Set parameters
    FPS = 10

    # Set capture device source
    video_src = 0
    signal_src = None

    # Set capture instance
    video_stream = VideoStream(src=video_src, fps=FPS, time_window=64)
    signal_stream = SignalStream(src=signal_src)

    # Process
    is_saving = False
    save_index = 0
    frames = []
    signal = []
    while True:
        # Show reference
        show_sin_signals(fps=FPS)

        # Get frame and signal data
        ret, frame = video_stream.read()
        signal_value = signal_stream.get_signal_value()

        if is_saving:
            frames.append(frame)
            signal.append(signal_value)
            show_signal('Signal', signal, 500)

        # Visualize
        frame = draw_fps(frame, fps=round(video_stream.get_fps()))
        cv2.imshow('Frame', frame)

        # Wait and key event
        key = cv2.waitKey(video_stream.delay())
        if key == 27:
            break
        elif key == ord('s'):
            if is_saving:
                print('> End saving.')

                xs = np.array(frames)
                ys = np.array(signal)
                print('> Shape of the data to be saved')
                print('Xs :', xs.shape, '/ Ys :', ys.shape)

                print('> Saving..')
                np.save(data_path('npy/frames%02d.npy'%save_index), xs)
                np.save(data_path('npy/signal%02d.npy' % save_index), ys)
                print('> Done.')
                frames = []
                signal = []
                save_index += 1
            else:
                print('> Start saving.')
            is_saving = not is_saving

    # Release resources
    cv2.destroyAllWindows()
    video_stream.release()
    signal_stream.close()
