import cv2
import numpy as np

from project_utils.resp_signal import SignalStream
from utils.camera.video_stream import VideoStream
from utils.visualization.signal import show_sin_signals, show_signal


class Experiment:

    def __init__(self, fps=10, name='hyeonsang', path='D:/respiration/npy/gui2/'):
        self.fps = fps
        self.name = name
        self.path = path

        self.reset_experiment()

    def reset_experiment(self):
        self.idx = 0
        self.bpm = 10
        self.frame_buffer = []
        self.reference_buffer = []
        self.sin_signal = []

        self.stop_state = 0

    def experiment_1(self, frame, signal_value):
        # Append
        self.frame_buffer.append(frame)
        self.reference_buffer.append(signal_value)
        data_len = len(self.frame_buffer)

        # Calc sin signal
        radian = self.idx * (2 * np.pi) / ((60.0 / self.bpm) * self.fps)

        self.sin_signal.append(np.sin(radian))

        if data_len > 100:
            del self.sin_signal[0]
            reference_signal = self.reference_buffer[-100: ]
        else:
            reference_signal = self.reference_buffer

        # Update data
        self.idx += 1
        if data_len > (20 * self.fps) and data_len % (10 * self.fps) == 0:
            prev_bpm = self.bpm
            self.bpm += (10 if data_len < ((20 + 40) * self.fps) else -10)
            self.idx = (self.idx * prev_bpm) / (self.bpm if self.bpm > 0 else 1)

        # Visualize
        show_signal('Sin', self.sin_signal, 500)
        show_signal('Reference', reference_signal, 500)

        if data_len >= self.fps * 90:
            print('Saving experiment 1..')

            np.save(self.path + '1_frame_' + self.name, self.frame_buffer)
            np.save(self.path + '1_signal_' + self.name, self.reference_buffer)

            print('Done.')
            self.reset_experiment()
            cv2.destroyWindow('Sin')
            cv2.destroyWindow('Reference')
            return False
        else:
            return True

    def experiment_2(self, frame, signal_value):
        # Append
        self.frame_buffer.append(frame)
        self.reference_buffer.append(signal_value)
        data_len = len(self.frame_buffer)

        # Calc sin signal
        radian = self.idx * (2 * np.pi) / ((60.0 / self.bpm) * self.fps)

        self.sin_signal.append(np.sin(radian))

        if data_len > 100:
            del self.sin_signal[0]
            reference_signal = self.reference_buffer[-100: ]
        else:
            reference_signal = self.reference_buffer

        # Update data
        self.idx += 1
        if self.stop_state == 0:
            pass
        elif self.stop_state == 1:
            self.idx -= 1
        elif self.stop_state == 2:
            pass
        elif self.stop_state == 3:
            self.idx -= 1

        if data_len > (20 * self.fps) and data_len % (5 * self.fps) == 0:
            if self.stop_state == 0:
                self.stop_state = 1
            elif self.stop_state == 1:
                self.stop_state = 2
            elif self.stop_state == 2:
                self.stop_state = 3
            elif self.stop_state == 3:
                self.stop_state = 0
                prev_bpm = self.bpm
                self.bpm += 10
                self.idx = (self.idx * prev_bpm) / self.bpm


        # Visualize
        show_signal('Sin', self.sin_signal, 500)
        show_signal('Reference', reference_signal, 500)

        if data_len >= self.fps * 80:
            print('Saving experiment 2..')

            np.save(self.path + '2_frame_' + self.name, self.frame_buffer)
            np.save(self.path + '2_signal_' + self.name, self.reference_buffer)

            print('Done.')
            self.reset_experiment()
            cv2.destroyWindow('Sin')
            cv2.destroyWindow('Reference')
            return False
        else:
            return True

    def experiment_3(self, frame, signal_value):
        # Append
        self.frame_buffer.append(frame)
        self.reference_buffer.append(signal_value)
        data_len = len(self.frame_buffer)

        if data_len > 100:
            reference_signal = self.reference_buffer[-100: ]
        else:
            reference_signal = self.reference_buffer

        # Visualize
        show_signal('Reference', reference_signal, 500)

        if data_len >= self.fps * 50:
            print('Saving experiment 3..')

            np.save(self.path + '3_frame_' + self.name, self.frame_buffer)
            np.save(self.path + '3_signal_' + self.name, self.reference_buffer)

            print('Done.')
            self.reset_experiment()
            cv2.destroyWindow('Reference')
            return False
        else:
            return True

    def experiment_4(self, frame, signal_value):
        # Append
        self.frame_buffer.append(frame)
        self.reference_buffer.append(signal_value)
        data_len = len(self.frame_buffer)

        if data_len > 100:
            reference_signal = self.reference_buffer[-100: ]
        else:
            reference_signal = self.reference_buffer

        # Visualize
        show_signal('Reference', reference_signal, 500)

        if data_len >= self.fps * 50:
            print('Saving experiment 4..')

            np.save(self.path + '4_frame_' + self.name, self.frame_buffer)
            np.save(self.path + '4_signal_' + self.name, self.reference_buffer)

            print('Done.')
            self.reset_experiment()
            cv2.destroyWindow('Reference')
            return False
        else:
            return True


def main():
    FPS = 10
    NAME = 'hyeonsang'

    stream = VideoStream(0, fps=FPS)
    signal_stream = SignalStream(src=None)
    exp = Experiment(FPS, name=NAME)
    state = 0

    while True:
        cv2.imshow("a", np.zeros((100,100, 3)))
        ret, frame = stream.read()
        signal_value = signal_stream.get_signal_value()
        if state == 1:
            ret = exp.experiment_1(frame, signal_value)
            if not ret:
                state = 0
        elif state == 2:
            ret = exp.experiment_2(frame, signal_value)
            if not ret:
                state = 0
        elif state == 3:
            ret = exp.experiment_3(frame, signal_value)
            if not ret:
                state = 0
        elif state == 4:
            ret = exp.experiment_4(frame, signal_value)
            if not ret:
                state = 0

        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
        if key == ord('1'):
            print('Experiment 1')
            state = 1
        elif key == ord('2'):
            print('Experiment 2')
            state = 2
        elif key == ord('3'):
            print('Experiment 3')
            state = 3
        elif key == ord('4'):
            print('Experiment 4')
            state = 4

    cv2.destroyAllWindows()
    stream.release()
    signal_stream.close()


if __name__ == '__main__':
    main()






