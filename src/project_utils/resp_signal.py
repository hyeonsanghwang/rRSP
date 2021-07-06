import cv2
import numpy as np

from vernier.process import GDX
from utils.visualization.signal import show_signal
from utils.camera.video_stream import VideoStream


class SignalStream:
    def __init__(self, src=None, window_size=64, window_period=None, skip_rate=1, show=True):
        self.window_size = window_size
        self.window_period = window_period
        self.skip_rate = skip_rate
        self.show = show
        if src is None:
            self.stream = GDX(window_size=window_size, window_period=window_period, is_usb=True, is_show=show)
            self.stream.start()
            self.data_available = True if self.stream.is_connected else False
        else:
            try:
                self.stream = None
                self.buffer = np.load(src)
                self.buffer_index = 0
                self.data_available = True
            except:
                self.data_available = False

    def get_signal_value(self):
        if self.stream is None:
            idx = self.buffer_index * self.skip_rate
            self.buffer_index += 1
            return self.buffer[idx]
        else:
            return self.stream.get_signal_value()

    def get_signal(self):
        if self.stream is None:
            idx = self.buffer_index * self.skip_rate
            window_size = self.window_size * self.skip_rate
            self.buffer_index += 1
            return self.buffer[max(0, idx - window_size):idx]
        else:
            return self.stream.get_signal()

    def set_window_period(self, sec):
        if self.stream is not None:
            self.stream.window_period = sec

    def close(self):
        if self.stream is not None:
            self.stream.close()


if __name__ == '__main__':
    # video_src = 'C:/Users/hyeon/Desktop/Develop/Respiration/data/001.mp4'
    video_src = 0

    # signal_src = '.npy'
    signal_src = None

    video_stream = VideoStream(src=video_src, fps=10, time_window=64)
    signal_stream = SignalStream(src=signal_src, window_size=200)

    signal = []
    while True:
        ret, frame = video_stream.read()
        signal_value = signal_stream.get_signal_value()
        signal.append(signal_value)
        if len(signal) > 100:
            del signal[0]
        if not ret:
            break

        cv2.imshow('frame', frame)
        show_signal('signal', signal, 500)
        key = cv2.waitKey(video_stream.delay())
        if key == 27:
            break
        print(video_stream.get_fps())

    cv2.destroyAllWindows()
    video_stream.release()
