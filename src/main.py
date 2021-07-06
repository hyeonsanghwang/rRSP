import os
import sys
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

from threading import Thread
from PyQt5.QtWidgets import QApplication

from gui.window import MainWindow, Parameter
from gui.tab_cluster import ClusterParameter
from gui.tab_optical import OpticalFlowParameter
from gui.tab_segmentation import SegmentationParameter
from gui.processing import ProcessManager

# from gui2 import Parameters

def start_gui(params):
    app = QApplication(sys.argv)
    window = MainWindow(params)
    window.show()
    app.exec_()


if __name__ == '__main__':
    params = Parameter()
    cluster_params = ClusterParameter()
    optical_params = OpticalFlowParameter()
    segmentation_params = SegmentationParameter()
    params_arr = [params, cluster_params, optical_params, segmentation_params]

    gui_thread = Thread(target=start_gui, args=(params_arr,))
    gui_thread.start()

    process = ProcessManager(params_arr)
    process.start()

    gui_thread.join()
