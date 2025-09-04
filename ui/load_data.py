import os
import json
from datetime import datetime
from PyQt6 import QtWidgets, QtGui, QtCore
import sys

class SavedDataWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Збережені дані")
        self.setWindowIcon(QtGui.QIcon('resources/images/icon.png'))
        self.setMinimumSize(500, 300)
        self.setStyleSheet(self.get_styles())
        self.parent = parent

        layout = QtWidgets.QVBoxLayout(self)

        title = QtWidgets.QLabel("Виберіть збережені дані:")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        # Список елементів
        self.list_widget = QtWidgets.QListWidget()
        self.load_saved_data()
        layout.addWidget(self.list_widget)

        # Кнопка вибору
        self.select_button = QtWidgets.QPushButton("Завантажити")
        self.select_button.clicked.connect(self.load_selected_data)
        layout.addWidget(self.select_button, alignment=QtCore.Qt.AlignmentFlag.AlignRight)

    def load_saved_data(self):
        # Ensure the directory exists
        save_dir = 'resources/saved_input'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            return

        # Clear existing items
        self.list_widget.clear()

        # Load all .json files from the directory
        for filename in os.listdir(save_dir):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(save_dir, filename), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        display_text = f"{data.get('function', 'N/A')} | {data.get('dimension', 'N/A')} | {data.get('date', 'N/A')}"
                        list_item = QtWidgets.QListWidgetItem(display_text)
                        list_item.setData(QtCore.Qt.ItemDataRole.UserRole, data)  # Store the full data
                        self.list_widget.addItem(list_item)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

    def load_selected_data(self):
        selected_items = self.list_widget.selectedItems()
        if not selected_items:
            QtWidgets.QMessageBox.warning(self, "Увага", "Будь ласка, виберіть елемент.")
            return

        selected_data = selected_items[0].data(QtCore.Qt.ItemDataRole.UserRole)
        self.loaded_data = selected_data
        self.accept()

    def get_styles(self):
        return """
        QWidget {
            font-family: 'Segoe UI', sans-serif;
            font-size: 14px;
            background-color: #f8f9fa;
            color: #2c3e50;
        }

        QListWidget {
            background-color: white;
            border: 1px solid #ced4da;
            border-radius: 6px;
            padding: 6px;
        }

        QListWidget::item {
            padding: 8px;
        }

        QListWidget::item:selected {
            background-color: #d0ebff;
            color: black;
        }

        QPushButton {
            background-color: #007bff;
            color: white;
            padding: 8px 16px;
            border-radius: 6px;
        }

        QPushButton:hover {
            background-color: #339af0;
        }

        QPushButton:pressed {
            background-color: #1c7ed6;
        }

        QLabel {
            margin-bottom: 8px;
        }
        """

# Запуск тестового вікна
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = SavedDataWindow()
    window.exec()