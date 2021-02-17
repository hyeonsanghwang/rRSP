import cv2

from resp_utils.decorator import synchronize
from utils.visualization.signal import show_sin_signals


class BPMManager:
    FRAME_NAME = "BPMs"

    def __init__(self, mutex):
        self.mutex = mutex
        self.bpms = []
        self.is_show = False
        self.init_bpms = False
        self.is_available = False

    @synchronize
    def set_parameters(self, bpms):
        self.bpms = bpms
        self.is_show = True
        self.init_bpms = True
        self.is_available = True

    @synchronize
    def show_bpms(self):
        if self.is_show:
            if self.is_available:
                show_sin_signals(name=self.FRAME_NAME, bpms=self.bpms, init_data=self.init_bpms)
                self.init_bpms = False
            else:
                print("[BPM Manager] Set parameters before calling show_bpms")

    @synchronize
    def stop(self):
        self.is_show = False
        cv2.destroyWindow(self.FRAME_NAME)
