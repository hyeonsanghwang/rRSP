
from PyQt5 import uic, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QWidget, QTabWidget

from project_utils.path import ui_path
from gui.tab_cluster import TabCluster
from gui.tab_optical import TabOpticalFlow
from gui.tab_segmentation import TabSegmentation

class WidgetSize:
    MARGIN = 5
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    SIGNAL_WIDTH = FRAME_WIDTH
    SIGNAL_HEIGHT = 150
    STATUS_WIDTH = FRAME_WIDTH
    STATUS_HEIGHT = 30
    NAVIGATE_WIDTH = 300
    NAVIGATE_HEIGHT_SHORT = FRAME_HEIGHT + MARGIN + SIGNAL_HEIGHT + STATUS_HEIGHT
    NAVIGATE_HEIGHT_LONG = NAVIGATE_HEIGHT_SHORT + SIGNAL_HEIGHT + MARGIN
    WINDOW_WIDTH = NAVIGATE_WIDTH + FRAME_WIDTH + MARGIN * 3
    WINDOW_HEIGHT_SHORT = NAVIGATE_HEIGHT_SHORT + MARGIN * 2
    WINDOW_HEIGHT_LONG = NAVIGATE_HEIGHT_LONG + MARGIN * 2


class WindowForm(QMainWindow, uic.loadUiType(ui_path('motion_ui.ui'))[0]):
    def __init__(self, params):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("rRSP")

        # Set widget size
        self.set_size()

        # Set initial fps
        self.spin_fps.setValue(params[0].fps)

        # Set tab (methods)
        self.tab_cluster = TabCluster(params[1])
        self.tab_mode.addTab(self.tab_cluster, 'Clustering')
        self.tab_optical = TabOpticalFlow(params[2])
        self.tab_mode.addTab(self.tab_optical, 'Optical flow')
        self.tab_segmentation = TabSegmentation(params[3])
        self.tab_mode.addTab(self.tab_segmentation, 'Segmentation')

        # Set bpms
        self.cb_bpms = [self.cb_bpm_5, self.cb_bpm_10, self.cb_bpm_15, self.cb_bpm_20, self.cb_bpm_25, self.cb_bpm_30,
                        self.cb_bpm_35, self.cb_bpm_40, self.cb_bpm_45, self.cb_bpm_50, self.cb_bpm_55, self.cb_bpm_60]
        # BPMs
        bpm_check_list = [False, True, True, True, True, True, True, True, False, False, False, False]
        for cb, check in zip(self.cb_bpms, bpm_check_list):
            cb.setChecked(check)

    def set_size(self):
        self.resize_window_short()

        status_x = WidgetSize.MARGIN
        status_y = WidgetSize.MARGIN
        label_frame_x = WidgetSize.MARGIN
        label_frame_y = WidgetSize.MARGIN + WidgetSize.STATUS_HEIGHT
        label_estimated_x = WidgetSize.MARGIN
        label_estimated_y = WidgetSize.MARGIN * 2 + WidgetSize.FRAME_HEIGHT + WidgetSize.STATUS_HEIGHT
        label_reference_x = WidgetSize.MARGIN
        label_reference_y = WidgetSize.MARGIN * 3 + WidgetSize.FRAME_HEIGHT + WidgetSize.SIGNAL_HEIGHT + WidgetSize.STATUS_HEIGHT
        widget_navigate_x = WidgetSize.MARGIN * 2 + WidgetSize.FRAME_WIDTH
        widget_navigate_y = WidgetSize.MARGIN

        self.widget_status.move(status_x, status_y)
        self.lbl_frame.move(label_frame_x, label_frame_y)
        self.widget_estimated.move(label_estimated_x, label_estimated_y)
        self.widget_reference.move(label_reference_x, label_reference_y)
        self.widget_navigate.move(widget_navigate_x, widget_navigate_y)

    def resize_window_short(self):
        self.widget_navigate.resize(WidgetSize.NAVIGATE_WIDTH, WidgetSize.NAVIGATE_HEIGHT_SHORT)
        self.setFixedSize(WidgetSize.WINDOW_WIDTH, WidgetSize.WINDOW_HEIGHT_SHORT)

    def resize_window_long(self):
        self.widget_navigate.resize(WidgetSize.NAVIGATE_WIDTH, WidgetSize.NAVIGATE_HEIGHT_LONG)
        self.setFixedSize(WidgetSize.WINDOW_WIDTH, WidgetSize.WINDOW_HEIGHT_LONG)


    def keyPressEvent(self, e: QtGui.QKeyEvent) -> None:
        if e.key() == Qt.Key_Escape:
            self.close()