import numpy as np
from keras.models import load_model

from resp_utils.decorator import synchronize


class ROIManager:
    def __init__(self, mutex):
        self.mutex = mutex

        self.model = None
        self.is_available = False
        self.score = None
        self.roi = None

    @synchronize
    def set_model(self, path):
        self.model = load_model(path, compile=False)
        self.is_available = True

    @synchronize
    def calculate_roi(self, data, threshold):
        if self.is_available:
            predict = self.model.predict(data)[0, ...]
            self.score = predict[..., 0]
            self.roi = (self.score >= threshold).astype(np.float32)
            return self.score, self.roi
        else:
            return None, None

    @synchronize
    def get_roi(self):
        if self.roi is not None:
            return (self.roi * 255).astype(np.uint8)
        else:
            return None

    @synchronize
    def get_score(self):
        if self.score is not None:
            return (self.score * 255).astype(np.uint8)
        else:
            return None

