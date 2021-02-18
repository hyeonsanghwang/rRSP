class Parameters:
    SIGNAL_CHANGED_FRAME = 0
    SIGNAL_CHANGED_ESTIMATED_SIGNAL = 1
    SIGNAL_CHANGED_REFERENCE_SIGNAL = 2
    SIGNAL_CHANGED_FPS = 3
    SIGNAL_START_PROCESS = 4
    SIGNAL_STOP_PROCESS = 5

    def __init__(self):
        # State
        self.is_started = True
        self.is_processing = False
        self.is_changed_parameters = False
        self.is_changed_source = False
        self.is_changed_model = False
        self.is_changed_mode_process = False
        self.is_show_bpm = False

        # Parameters
        self.fps = 20
        self.process_fps = 5
        self.window_size = 64
        self.resize_ratio = 8
        self.color_domain = 0
        self.detect_threshold = 0.5

        # Data source
        self.src_type = None
        self.src_video = None
        self.src_signal = None
        self.src_model = None

        # Mode
        self.mode_process = 0
        self.mode_show = 0

        # BPMs
        self.bpm_list = []

        # Signals
        self.signal_dict = dict()

    def set_start(self):
        self.is_processing = True
        self.is_changed_parameters = True
        self.is_changed_source = True
        self.is_changed_mode_process = True

        self.src_type = 0
        self.src_video = 0
        self.src_signal = None

    def set_signal(self, key, value):
        self.signal_dict[key] = value

    def signal(self, key):
        return self.signal_dict[key]
