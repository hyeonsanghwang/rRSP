import cv2
import numpy as np


class OpticalflowTester:
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def __init__(self, params):
        self.params = params
        self.resize_ratio = 0
        self.buffer_size = 0
        self.estimated_signal = None
        self.result_frame = None

        # Optical flow variables
        self.gray = None
        self.prev_gray = None

        self.prev_points = None
        self.curr_points = None
        self.point_status = None

    def process(self, frame, roi, max_point=50):
        self.frame = frame
        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)

        if self.prev_gray is None:
            self.prev_gray = self.gray
            return self.estimated_signal

        # Calculate optical flow
        ys, xs = np.where(roi == 1.0)
        xs = np.expand_dims(xs, axis=-1)
        ys = np.expand_dims(ys, axis=-1)
        ps = np.append(xs, ys, axis=-1).astype(np.float32) * self.resize_ratio + self.resize_ratio / 2

        # Random sampling
        point_num = min(max_point, ps.shape[0])
        idx = np.random.choice(ps.shape[0], point_num, replace=False)
        ps = ps[idx, :]

        self.prev_points = np.expand_dims(ps, axis=1)
        self.curr_points, self.point_status, e = cv2.calcOpticalFlowPyrLK(self.prev_gray,
                                                                          self.gray,
                                                                          self.prev_points,
                                                                          None,
                                                                          **self.lk_params)
        # Set results
        self.prev_points = self.prev_points[self.point_status == 1.0]
        self.curr_points = self.curr_points[self.point_status == 1.0]
        self.prev_gray = self.gray

        # Calculate signal value
        movement = self.curr_points - self.prev_points
        binary_movement = ((movement > 0).astype(np.float32) * 2) - 1
        mean_movement = binary_movement.mean(axis=0)

        value = 0 if mean_movement.size == 0 else mean_movement[1]

        if self.estimated_signal is None:
            self.estimated_signal = [0]
        self.estimated_signal.append(self.estimated_signal[-1] - value)
        if len(self.estimated_signal) > self.buffer_size:
            del self.estimated_signal[0]

        return self.estimated_signal

    def draw_optical_flow_result(self):
        show_frame = self.frame.copy()
        mask = np.zeros_like(show_frame)
        for i, (curr, prev) in enumerate(zip(self.curr_points, self.prev_points)):
            x1, y1 = prev.ravel()
            x2, y2 = curr.ravel()
            mask = cv2.line(mask, (x2, y2), (x1, y1), (0, 255, 255), 2)
            show_frame = cv2.circle(show_frame, (x2, y2), 3, (0, 255, 255), -1)
        img = cv2.add(show_frame, mask)
        return img

    def stop(self):
        pass
