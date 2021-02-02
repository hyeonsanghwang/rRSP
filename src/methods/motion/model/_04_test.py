import cv2

from utils.capture import VideoStream, SignalStream
from utils.path import result_path, data_path
from utils.visualize import show_bpms


from motion.data._00_data_preprocess import ColorDomain
from motion.model._04_1_clustering_test import ClusteringTester
# from motion.model._04_2_optical_flow_test import OpticalFlowTester
from motion.model._04_3_test_revision import OpticalFlowTester


if __name__ == '__main__':
    # Set process parameters
    FPS = 60
    PROCESS_FPS = 5
    WINDOW_SIZE = 64
    RESIZE_RATIO = 24
    COLOR_DOMAIN = ColorDomain.RGB
    MODEL_PATH = result_path('detect_roi/model_fc.h5')
    ROI_THRESHOLD = 0.5
    USE_SENSOR = True

    # Set video stream
    video_src = data_path('video.mp4')
    # video_src = 'E:/data/temp/1_1_frames.npy'
    # video_src = 0
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
