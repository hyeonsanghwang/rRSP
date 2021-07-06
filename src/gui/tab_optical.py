import os

from PyQt5 import uic, QtGui
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QWidget, QFileDialog

from methods.roi_detection.data._00_data_preprocess import ColorDomain
from project_utils.path import ui_path


class OpticalFlowParameter:
    SHOW_MODE_ORIGINAL = 0
    SHOW_MODE_RESIZED = 1
    SHOW_MODE_SCORE = 2
    SHOW_MODE_ROI = 3
    SHOW_MODE_OPTICAL = 4

    SIGNAL_CHANGE_PROGRESS = 0

    def __init__(self):
        # Process parameters
        self.parameter_changed = False
        self.process_fps = 5
        self.window_size = 64
        self.resize_ratio = 8
        self.color_domain = ColorDomain.RGB
        self.threshold = 0.5

        # ROI detector
        self.model_changed = False
        self.model_path = '../model/detect_roi/model.h5'

        # Show mode
        self.show_mode_changed = False
        self.show_mode = self.SHOW_MODE_ORIGINAL

        # Shared signals
        self.signal_change_progress = None

    def set_signals(self, signal_type, signal):
        if signal_type == self.SIGNAL_CHANGE_PROGRESS:
            self.signal_change_progress = signal


class TabOpticalFlow(QWidget, uic.loadUiType(ui_path('tab_optical.ui'))[0]):
    """
    [Widgets]
    - Parameters
        spin_process_fps
        spin_window_size
        spin_resize_ratio
        combo_color_domain
        spin_threshold
        b_set_parameter
    - ROI detect model
        lbl_model_name
        b_set_model
    - Show mode
        b_show_original
        b_show_resized
        b_show_score
        b_show_roi
        b_flip
        b_show_cluster
        b_show_symmetry_cluster
    - Progress bar
        progress_bar
    """
    signal_changed_progress = pyqtSignal(int, int)

    def __init__(self, params: OpticalFlowParameter):
        super().__init__()
        self.setupUi(self)
        self.params = params

        # Init widgets
        self._init_widgets()
        self._set_model_path(params.model_path)

        # Connect event
        self.b_set_parameter.clicked.connect(self._on_clicked_set_parameter)
        self.b_set_model.clicked.connect(self._on_clicked_set_model)

        self.b_show_original.clicked.connect(lambda: self._on_clicked_show_mode(self.b_show_original))
        self.b_show_resized.clicked.connect(lambda: self._on_clicked_show_mode(self.b_show_resized))
        self.b_show_score.clicked.connect(lambda: self._on_clicked_show_mode(self.b_show_score))
        self.b_show_roi.clicked.connect(lambda: self._on_clicked_show_mode(self.b_show_roi))
        self.b_show_optical.clicked.connect(lambda: self._on_clicked_show_mode(self.b_show_optical))

        # Connect custom signal
        self.signal_changed_progress.connect(self.change_progress)
        self.params.set_signals(OpticalFlowParameter.SIGNAL_CHANGE_PROGRESS, self.signal_changed_progress)

        # Init parameters
        self._set_parameters()

    # ----------------------------------------------------------------------
    #                          Initialize widgets
    # ----------------------------------------------------------------------
    def _init_widgets(self):
        # Color model combo box
        self.combo_color_domain.addItem("RGB")
        self.combo_color_domain.addItem("Gray")
        self.combo_color_domain.addItem("YCrCb")
        self.combo_color_domain.addItem("HSV")

        # Parameters
        self.spin_process_fps.setValue(self.params.process_fps)
        self.spin_window_size.setValue(self.params.window_size)
        self.spin_resize_ratio.setValue(self.params.resize_ratio)
        self.combo_color_domain.setCurrentIndex(self.params.color_domain)
        self.spin_threshold.setValue(self.params.threshold)

    # ----------------------------------------------------------------------
    #                          Widget event listener
    # ----------------------------------------------------------------------
    def _on_clicked_set_parameter(self):
        print('꺆')
        self._set_parameters()

    def _on_clicked_set_model(self):
        path = QFileDialog.getOpenFileName(self,
                                           'Load model path',
                                           '../model/detect_roi/',
                                           filter='Keras model (*.h5)')[0]
        if path:
            self._set_model_path(path)

    def _on_clicked_show_mode(self, widget):
        if widget == self.b_show_original:
            self.params.show_mode = OpticalFlowParameter.SHOW_MODE_ORIGINAL
        elif widget == self.b_show_resized:
            self.params.show_mode = OpticalFlowParameter.SHOW_MODE_RESIZED
        elif widget == self.b_show_score:
            self.params.show_mode = OpticalFlowParameter.SHOW_MODE_SCORE
        elif widget == self.b_show_roi:
            self.params.show_mode = OpticalFlowParameter.SHOW_MODE_ROI
        elif widget == self.b_show_optical:
            self.params.show_mode = OpticalFlowParameter.SHOW_MODE_OPTICAL
        self.params.show_mode_changed = True

    def _on_clicked_function(self, widget):
        if widget == self.b_flip:
            self.params.flip = not self.params.flip
        elif widget == self.b_show_cluster:
            self.params.show_cluster = not self.params.show_cluster
        elif widget == self.b_show_symmetry_cluster:
            self.params.show_sym_cluster = not self.params.show_sym_cluster
        self.params.function_changed = True

    # -----------------------------------------------------------------------------------------
    #                                      Signal processing
    # -----------------------------------------------------------------------------------------
    @pyqtSlot(int, int)
    def change_progress(self, curr, max_val):
        if curr == 1:
            self.progress_bar.setRange(1, max_val-1)
        self.progress_bar.setValue(curr)

    # ----------------------------------------------------------------------
    #                               Set methods
    # ----------------------------------------------------------------------
    def _set_parameters(self):
        self.params.process_fps = self.spin_process_fps.value()
        self.params.window_size = self.spin_window_size.value()
        self.params.resize_ratio = self.spin_resize_ratio.value()
        self.params.color_domain = self.combo_color_domain.currentIndex()
        self.params.threshold = self.spin_threshold.value()
        self.params.parameter_changed = True

    def _set_model_path(self, path):
        name = os.path.split(path)[-1]
        self.lbl_model_name.setText(name)
        self.params.model_path = path
        self.params.model_changed = True
