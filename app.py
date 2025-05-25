import sys
from PyQt6 import QtWidgets

from ui.main import MainWindow
from tests.logic import optimization_tests


def run_program():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    optimization_tests(type="ND")