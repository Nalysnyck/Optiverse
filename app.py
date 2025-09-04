import sys
from PyQt6 import QtWidgets

from ui.main import MainWindow

def run_program():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    run_program()