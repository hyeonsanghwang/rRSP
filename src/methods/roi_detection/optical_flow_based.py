import numpy as np
import cv2
from numba import cuda

from keras.models import load_model

from methods.roi_detection.data._00_data_preprocess import frame_process
from utils.visualization.signal import show_signal
from gui.tab_optical import OpticalFlowParameter

class OpticalFlowBased:
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def __init__(self, params: OpticalFlowParameter, capture_fps):
        self.params = params

        self.capture_fps = capture_fps
        self.model_path = params.model_path
        self.maximum_point = 100

        self.set_parameters(params.window_size,
                            params.process_fps,
                            params.color_domain,
                            params.resize_ratio,
                            params.threshold)

        # ROI detection model
        self.roi_model = None

        # buffers
        self.buffer = []
        self.signal = [0]
        self.return_signal = np.array([0] * self.window_size, np.float32)

        # Single image buffer
        self.frame = None
        self.processed = None
        self.score = None
        self.roi = None
        self.estimated_signal = None

        # Single image buffer (for optical flow)
        self.gray = None
        self.prev_gray = None

        # ETC
        self.show_mode = 0

    # -----------------------------------------------------------------------------------------------
    #                                          Set parameters
    # -----------------------------------------------------------------------------------------------
    def set_parameters(self, window_size, process_fps, color_domain, resize_ratio, roi_threshold):
        self.window_size = window_size
        self.process_fps = process_fps
        self.color_domain = color_domain
        self.resize_ratio = resize_ratio
        self.roi_threshold = roi_threshold

        # Select index
        fps_ratio = (self.capture_fps / process_fps)
        self.buffer_size = int(fps_ratio * window_size)
        index = (np.array(list(range(self.buffer_size))) / fps_ratio).astype(np.int)
        self.target_index = (index != np.insert(index, 0, -1)[:-1])

    def set_model_path(self, model_path):
        self.model_path = model_path
        # cuda.select_device(0)
        # cuda.close()
        self.roi_model = None

    # -----------------------------------------------------------------------------------------------
    #                                   Reset memory and buffers
    # -----------------------------------------------------------------------------------------------
    def reset(self):
        # Release model
        # cuda.select_device(0)
        # cuda.close()
        self.roi_model = None

        # Reset buffers
        self.buffer = []
        self.signal = [0]
        self.frame = None
        self.processed = None
        self.score = None
        self.roi = None
        self.gray = None
        self.prev_gray = None
        self.prev_points = None
        self.curr_points = None

    # -----------------------------------------------------------------------------------------------
    #                                        Main processing
    # -----------------------------------------------------------------------------------------------
    def get_respiration(self, frame):
        # Save as member variable
        self.frame = frame
        self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = self.gray
            return self.return_signal

        # Load ROI detection model
        if self.roi_model is None:
            self.roi_model = load_model(self.model_path, compile=False)

        # Calculate ROI
        frames = self._get_frames(frame)
        if frames is None:
            return self.return_signal
        self.score, self.roi = self._get_roi(frames)

        # Calculate ROI points
        ps = self._get_roi_points(self.roi)
        if ps.shape[0] == 0:
            return self.return_signal

        # Calculate optical flow
        self.prev_points, self.curr_points = self._get_optical_flow(ps)

        # Estimate motion
        movement = self._get_motion_information(self.prev_points, self.curr_points)

        # Set signal
        self.signal.append(self.signal[-1] - movement)
        if len(self.signal) > self.buffer_size:
            del self.signal[0]

        return np.array(self.signal, np.float32)

    def get_show_frame(self):
        if self.show_mode == OpticalFlowParameter.SHOW_MODE_ORIGINAL:
            return self.frame

        elif self.show_mode == OpticalFlowParameter.SHOW_MODE_RESIZED:
            return (self.processed * 255).astype(np.uint8)

        elif self.show_mode == OpticalFlowParameter.SHOW_MODE_SCORE:
            return self.frame if self.score is None else (self.score * 255).astype(np.uint8)

        elif self.show_mode == OpticalFlowParameter.SHOW_MODE_ROI:
            return self.frame if self.roi is None else (self.roi * 255).astype(np.uint8)

        elif self.show_mode == OpticalFlowParameter.SHOW_MODE_OPTICAL:
            if self.prev_points is None or self.curr_points is None:
                return self.frame
            else:
                show_frame = self.frame.copy()
                mask = np.zeros_like(show_frame)
                for i, (curr, prev) in enumerate(zip(self.curr_points, self.prev_points)):
                    x1, y1 = prev.ravel()
                    x2, y2 = curr.ravel()
                    mask = cv2.line(mask, (x2, y2), (x1, y1), (0, 255, 255), 2)
                    show_frame = cv2.circle(show_frame, (x2, y2), 3, (0, 255, 255), -1)
                return cv2.add(show_frame, mask)

    # -----------------------------------------------------------------------------------------------
    #                                 Methods for extracting signal
    # -----------------------------------------------------------------------------------------------
    def _get_frames(self, frame):
        # Process frame
        self.processed = frame_process(frame, self.color_domain, self.resize_ratio) / 255.0
        self.buffer.append(self.processed)

        # Check buffer length
        if len(self.buffer) < self.buffer_size:
            self.params.signal_change_progress.emit(len(self.buffer), self.buffer_size)
            return None
        elif len(self.buffer) == self.buffer_size + 1:
            del self.buffer[0]
        elif len(self.buffer) > self.buffer_size + 1:
            self.buffer = self.buffer[-self.buffer_size:]

        # Get target frames
        np_buffer = np.array(self.buffer, np.float32)
        frames = np_buffer[self.target_index]
        return frames

    def _get_roi(self, frames):
        pred = self.roi_model.predict(np.expand_dims(frames, 0))[0, ...]
        score = pred[..., 0]
        roi = (score >= self.roi_threshold).astype(np.float32)
        return score, roi

    def _get_roi_points(self, roi):
        # Calculate ROI points
        ys, xs = np.where(roi == 1.0)
        xs = np.expand_dims(xs, axis=-1)
        ys = np.expand_dims(ys, axis=-1)
        ps = np.append(xs, ys, axis=-1).astype(np.float32) * self.resize_ratio + self.resize_ratio / 2

        # Random sampling
        point_num = min(self.maximum_point, ps.shape[0])
        idx = np.random.choice(ps.shape[0], point_num, replace=False)
        ps = ps[idx, :]
        return ps

    def _get_optical_flow(self, ps):
        prev_points = np.expand_dims(ps, axis=1)
        curr_points, status, e = cv2.calcOpticalFlowPyrLK(self.prev_gray,
                                                          self.gray,
                                                          prev_points,
                                                          None,
                                                          **self.lk_params)
        prev_points = prev_points[status == 1.0]
        curr_points = curr_points[status == 1.0]
        self.prev_gray = self.gray
        return prev_points, curr_points

    def _get_motion_information(self, prev_points, curr_points):
        movement = curr_points - prev_points
        # binary_movement = ((movement > 0).astype(np.float32) * 2) - 1
        mean_movement = movement.mean(axis=0)

        value = 0 if mean_movement.size == 0 else mean_movement[1]
        return value


if __name__ == '__main__':
    model = OpticalFlowBased(64, 20, 5, 0, 8, 0.5, '../../../model/detect_roi/model.h5')

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        ret, frame = cap.read()
        cv2.imshow('Frame', frame)

        signal = model.get_respiration(frame)
        show_signal('signal', signal, 500)

        key = cv2.waitKey(1)
        if key == 27:
            break
    cv2.destroyAllWindows()
