import os

from PyQt5 import uic, QtGui
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QWidget, QFileDialog

from methods.roi_detection.data._00_data_preprocess import ColorDomain
from project_utils.path import ui_path


class SegmentationParameter:
    SHOW_MODE_ORIGINAL = 0
    SHOW_MODE_SEGMENTATION = 1
    SHOW_MODE_MASK = 2
    SHOW_MODE_MOTION = 3

    def __init__(self):
        # Show mode
        self.show_mode_change = False
        self.show_mode = self.SHOW_MODE_ORIGINAL


class TabSegmentation(QWidget, uic.loadUiType(ui_path('tab_segmentation.ui'))[0]):
    """
    [Widgets]
    - Parameters
        spin_example
        b_set_parameter
    - Show mode
        b_show_original
        b_show_segmentation
        b_show_mask
        b_show_motion
    """

    def __init__(self, params: SegmentationParameter):
        super().__init__()
        self.setupUi(self)
        self.params = params

        # Connect event
        self.b_show_original.clicked.connect(lambda: self._on_clicked_show_mode(self.b_show_original))
        self.b_show_segmentation.clicked.connect(lambda: self._on_clicked_show_mode(self.b_show_segmentation))
        self.b_show_mask.clicked.connect(lambda: self._on_clicked_show_mode(self.b_show_mask))
        self.b_show_motion.clicked.connect(lambda: self._on_clicked_show_mode(self.b_show_motion))

    def _on_clicked_show_mode(self, button):
        if button == self.b_show_original:
            self.params.show_mode = SegmentationParameter.SHOW_MODE_ORIGINAL
        elif button == self.b_show_segmentation:
            self.params.show_mode = SegmentationParameter.SHOW_MODE_SEGMENTATION
        elif button == self.b_show_mask:
            self.params.show_mode = SegmentationParameter.SHOW_MODE_MASK
        elif button == self.b_show_motion:
            self.params.show_mode = SegmentationParameter.SHOW_MODE_MOTION
        self.params.show_mode_change = True
