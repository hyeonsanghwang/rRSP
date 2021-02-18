import numpy as np

from resp_utils.decorator import synchronize
from methods.motion.data._00_data_preprocess import frame_process


class FrameManager:
    def __init__(self, lock):
        self.mutex = lock
        self.resize_ratio = None
        self.color = None
        self.fps = None
        self.process_fps = None
        self.window_size = None
        self.buffer_size = None
        self.target_index = None
        self.frame = None
        self.processed = None
        self.buffer = None
        self.frames = None

        self.is_available = False

    def __clear_buffer(self):
        self.buffer = []
        self.frames = None

    @synchronize
    def set_parameters(self, resize_ratio, color, fps, process_fps, window_size):
        self.resize_ratio = resize_ratio
        self.color = color
        self.fps = fps
        self.process_fps = process_fps
        self.window_size = window_size

        fps_ratio = (fps / process_fps)
        self.buffer_size = int(fps_ratio * window_size)
        index = (np.array(list(range(self.buffer_size))) / fps_ratio).astype(np.int)
        self.target_index = (index != np.insert(index, 0, -1)[:-1])

        self.__clear_buffer()
        self.is_available = True

    @synchronize
    def clear_buffer(self):
        self.__clear_buffer()

    @synchronize
    def set_frame(self, frame):
        if self.is_available:
            self.processed = frame_process(frame, self.color, self.resize_ratio) / 255.0
            self.buffer.append(self.processed)

            if len(self.buffer) < self.buffer_size:
                return
            elif len(self.buffer) > self.buffer_size:
                del self.buffer[0]

            xs = np.array([self.buffer], np.float32)
            self.frames = xs[:, self.target_index]
        else:
            print("[Frame Manager] Set parameters before calling set_frame")

    @synchronize
    def get_frames(self):
        return self.frames

