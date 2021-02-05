
from PyQt5 import uic, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow

from path import ui_path


class WidgetSize:
    MARGIN = 5
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    SIGNAL_WIDTH = FRAME_WIDTH
    SIGNAL_HEIGHT = 150
    NAVIGATE_WIDTH = 300
    NAVIGATE_HEIGHT_SHORT = FRAME_HEIGHT + MARGIN + SIGNAL_HEIGHT
    NAVIGATE_HEIGHT_LONG = NAVIGATE_HEIGHT_SHORT + SIGNAL_HEIGHT + MARGIN
    WINDOW_WIDTH = NAVIGATE_WIDTH + FRAME_WIDTH + MARGIN * 3
    WINDOW_HEIGHT_SHORT = NAVIGATE_HEIGHT_SHORT + MARGIN * 2
    WINDOW_HEIGHT_LONG = NAVIGATE_HEIGHT_LONG + MARGIN * 2


class WindowForm(QMainWindow, uic.loadUiType(ui_path('motion_ui.ui'))[0]):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("rRSP")

        self.set_color_domain()
        self.set_size()

        self.cb_bpms = [self.cb_bpm_5, self.cb_bpm_10, self.cb_bpm_15, self.cb_bpm_20, self.cb_bpm_25, self.cb_bpm_30,
                        self.cb_bpm_35, self.cb_bpm_40, self.cb_bpm_45, self.cb_bpm_50, self.cb_bpm_55, self.cb_bpm_60]

    def set_color_domain(self):
        self.combo_color_domain.addItem("RGB")
        self.combo_color_domain.addItem("Gray")
        self.combo_color_domain.addItem("YCrCb")
        self.combo_color_domain.addItem("HSV")

    def set_size(self):
        self.resize_window_short()
        self.widget_navigate.move(WidgetSize.MARGIN, WidgetSize.MARGIN)
        self.lbl_frame.move(WidgetSize.MARGIN * 2 + WidgetSize.NAVIGATE_WIDTH, WidgetSize.MARGIN)
        self.widget_estimated.move(WidgetSize.MARGIN * 2 + WidgetSize.NAVIGATE_WIDTH,
                                   WidgetSize.MARGIN * 2 + WidgetSize.FRAME_HEIGHT)
        self.widget_reference.move(WidgetSize.MARGIN * 2 + WidgetSize.NAVIGATE_WIDTH,
                                   WidgetSize.MARGIN * 3 + WidgetSize.FRAME_HEIGHT + WidgetSize.SIGNAL_HEIGHT)

    def resize_window_short(self):
        self.widget_navigate.resize(WidgetSize.NAVIGATE_WIDTH, WidgetSize.NAVIGATE_HEIGHT_SHORT)
        self.setFixedSize(WidgetSize.WINDOW_WIDTH, WidgetSize.WINDOW_HEIGHT_SHORT)

    def resize_window_long(self):
        self.widget_navigate.resize(WidgetSize.NAVIGATE_WIDTH, WidgetSize.NAVIGATE_HEIGHT_LONG)
        self.setFixedSize(WidgetSize.WINDOW_WIDTH, WidgetSize.WINDOW_HEIGHT_LONG)

    def keyPressEvent(self, e: QtGui.QKeyEvent) -> None:
        if e.key() == Qt.Key_Escape:
            self.close()