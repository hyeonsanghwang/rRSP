import threading
import numpy as np
import cv2
from time import perf_counter

from vernier.gdx import gdx
from utils.visualization.signal import signal_to_frame
from utils.visualization.common import draw_fps


class GDX:
    BUFFER_SIZE = 1024000

    def __init__(self, window_size=1000, window_period=None, is_usb=True, is_show=True, frame_width=500, frame_height=100):
        self.window_size = window_size
        self.window_period = window_period
        self.is_usb = is_usb
        self.is_show = is_show
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.thread = None
        self.gdx = None

        self.init_variables()

    def init_variables(self):
        # Respiration state
        self.buffer = []
        self.time_stamp = []
        self.bpm = 0

        # Flag parameters
        self.is_started = False

    def start(self):
        self.is_started = True
        self.gdx = gdx()
        if self.is_usb:
            self.gdx.open_usb()
        else:
            self.gdx.open_ble()

        self.gdx.select_sensors()
        self.gdx.start(period=10)

        self.thread = threading.Thread(target=self._start)
        self.thread.start()

    def _start(self):
        while self.is_started:
            self.time_stamp.append(perf_counter())
            val, bpm = self.gdx.read()
            self.bpm = self.bpm if np.isnan(bpm) else bpm
            self.buffer.append(val)

            if len(self.buffer) > self.BUFFER_SIZE:
                del self.buffer[0]
                del self.time_stamp[0]

            if self.is_show:
                start_index = self._get_start_index()
                window_fps = self._get_window_fps()
                signal = self.buffer[start_index: ]

                frame = signal_to_frame(signal, width=self.frame_width, height=self.frame_height)
                cv2.putText(frame, '%02d fps' % round(window_fps), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(frame, '%02d bpm' % self.bpm, (self.frame_width - 70, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.imshow('Vernier GDX signal', frame)

                key = cv2.waitKey(1)
                if key == 27:
                    self.is_started = False

    def _get_start_index(self):
        if self.window_period is None:
            start_index = max(0, len(self.buffer) - self.window_size)
        else:
            time_stamp = np.array(self.time_stamp)
            time_diff = time_stamp - (time_stamp[-1] - self.window_period)
            start_index = np.fabs(time_diff).argmin()
        return start_index

    def _get_window_fps(self):
        start_index = self._get_start_index()
        times = self.time_stamp[start_index: ]
        if times[0] == times[-1]:
            fps = 0
        else:
            fps = len(times) / (times[-1] - times[0])
        return fps

    def get_signal_value(self):
        value = 0 if len(self.buffer) == 0 else self.buffer[-1]
        return value

    def get_signal(self):
        start_index = self._get_start_index()
        return self.buffer[start_index: ]

    def get_bpm(self):
        return self.bpm

    def close(self):
        self.is_started = False
        if self.thread is not None:
            # Wait until the thread ends
            self.thread.join()

            # Sensor close
            if self.gdx is not None:
                self.gdx.stop()
                self.gdx.close()
                self.gdx = None

            # Clear buffers
            self.buffer = []
            self.sync_list = []
