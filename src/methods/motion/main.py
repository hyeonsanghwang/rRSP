import os
import sys
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

from threading import Thread
from PyQt5.QtWidgets import QApplication

from methods.motion.test.window import MainWindow
from methods.motion.test.processing import ProcessManager
from methods.motion.test.module_parameters import Parameters


def start_gui(params):
    app = QApplication(sys.argv)
    window = MainWindow(params)
    window.show()
    app.exec_()


if __name__ == '__main__':
    params = Parameters()

    gui_thread = Thread(target=start_gui, args=(params, ))
    gui_thread.start()

    process = ProcessManager(params)
    process.start()

    gui_thread.join()
