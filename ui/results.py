import sys
from PyQt6 import QtWidgets, QtGui, QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import json
import os
from datetime import datetime
from ui.method_results import IterationViewer
import random

def random_color():
    return "#" + ''.join(random.choices('0123456789ABCDEF', k=6))

class ResultsWindow(QtWidgets.QDialog):
    def __init__(self, input_data, optimization_type, results):
        super().__init__()
        self.setWindowTitle("Результати оптимізації")
        self.setWindowIcon(QtGui.QIcon('resources/images/icon.png'))
        self.setMinimumSize(1337, 500)
        self.setStyleSheet(self.get_styles())

        self._dragging = False
        self._last_mouse_pos_px = None
        self.point_artists = {}
        
        self.input_data = input_data
        self.optimization_type = optimization_type
        self.results = results
        # Generate a random color for each method
        self.method_colors = {method: random_color() for method in self.results.keys()}

        main_layout = QtWidgets.QHBoxLayout(self)

        # Графік
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)

        self.plot_function_with_points()
        main_layout.addWidget(self.canvas, stretch=3)

        # Права частина
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)

        label = QtWidgets.QLabel("Методи оптимізації:")
        label.setStyleSheet("font-size: 16px; font-weight: bold;")
        right_layout.addWidget(label)

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            "Назва методу", "Успішно", "К-сть ітерацій", "Точка оптимуму", "Час виконання, с"
        ])
        self.table.setSortingEnabled(True)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.table.setMouseTracking(True)
        self.table.viewport().installEventFilter(self)
        self.table.cellDoubleClicked.connect(self.open_method_results)

        self.fill_table()
        self.set_column_widths()
        right_layout.addWidget(self.table)

        self.save_button = QtWidgets.QPushButton("Зберегти вхідні дані")
        self.save_button.clicked.connect(self.save_input_data)
        right_layout.addWidget(self.save_button, alignment=QtCore.Qt.AlignmentFlag.AlignRight)

        main_layout.addWidget(right_panel, stretch=4)

    def compute_function(self, array):
        # Використовуємо функцію з input_data
        if self.optimization_type == "1D":
            return [self.input_data["function"]([x]) for x in array]
        else:
            # Для ND випадку, x має бути масивом
            return [self.input_data["function"](x) for x in array]

    def fill_table(self):
        self.table.setRowCount(len(self.results))
        for row, (method_name, result) in enumerate(self.results.items()):
            color = self.method_colors.get(method_name, "#333")
            name_item = QtWidgets.QTableWidgetItem(method_name)
            name_item.setForeground(QtGui.QColor(color))
            self.table.setItem(row, 0, name_item)
            
            # Успішно
            success_item = QtWidgets.QTableWidgetItem("Так" if result['is_successful'] else "Ні")
            success_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 1, success_item)
            
            # Кількість ітерацій
            iterations = len(result['iterations'])
            iter_item = QtWidgets.QTableWidgetItem(str(iterations))
            iter_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 2, iter_item)
            
            # Оптимум
            if result['is_successful']:
                if self.optimization_type == "1D":
                    optimum = f"{result['optimum']['position']['x']:.5f}"
                else:
                    optimum = str(result['optimum']['position'])
            else:
                optimum = "-"
            opt_item = QtWidgets.QTableWidgetItem(optimum)
            opt_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 3, opt_item)
            
            # Час виконання
            time_item = QtWidgets.QTableWidgetItem(f"{result['time']:.6f}")
            time_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 4, time_item)

    def set_column_widths(self):
        self.table.setColumnWidth(0, 200)  # Назва
        self.table.setColumnWidth(1, 80)   # Успішно
        self.table.setColumnWidth(2, 120)  # Ітерації
        self.table.setColumnWidth(3, 150)  # Точка
        self.table.setColumnWidth(4, 150)  # Час

    def plot_function_with_points(self):
        self.ax.clear()
        
        if self.optimization_type == "1D":
            x = np.linspace(self.input_data["interval"][0], self.input_data["interval"][1], 1000)
            y = self.compute_function(x)
            self.ax.plot(x, y, label="f(x)", color="#2c3e50")

            self.point_artists.clear()
            for method_name, result in self.results.items():
                if result['is_successful']:
                    x_opt = result['optimum']['position']['x']
                    y_opt = result['optimum']['value']
                    color = self.method_colors.get(method_name, "#333")
                    artist, = self.ax.plot(x_opt, y_opt, 'o', label=method_name, color=color, markersize=8)
                    self.point_artists[method_name] = artist

            self.ax.set_title(f"Функція {self.input_data['expression']}")
            self.ax.set_xlabel("x")
            self.ax.set_ylabel("f(x)")
        
        self.ax.grid(True)
        self.canvas.draw()

    def eventFilter(self, source, event):
        if source == self.table.viewport() and event.type() == QtCore.QEvent.Type.MouseMove:
            index = self.table.indexAt(event.pos())
            if index.isValid():
                row = index.row()
                method_name = self.table.item(row, 0).text()
                self.highlight_method_point(method_name)
        return super().eventFilter(source, event)

    def highlight_method_point(self, method_name):
        for name, artist in self.point_artists.items():
            if name == method_name:
                artist.set_alpha(1.0)
                artist.set_markersize(10)
            else:
                artist.set_alpha(0.2)
                artist.set_markersize(6)
        self.canvas.draw_idle()

    def on_press(self, event):
        if event.button == 1 and event.inaxes:
            self._dragging = True
            self._last_mouse_pos_px = (event.x, event.y)

    def on_release(self, event):
        self._dragging = False
        self._last_mouse_pos_px = None

    def on_motion(self, event):
        if self._dragging and event.inaxes and self._last_mouse_pos_px:
            dx_px = event.x - self._last_mouse_pos_px[0]
            dy_px = event.y - self._last_mouse_pos_px[1]

            dx_data, dy_data = self.ax.transData.inverted().transform((dx_px, dy_px)) - \
                            self.ax.transData.inverted().transform((0, 0))

            x0, x1 = self.ax.get_xlim()
            y0, y1 = self.ax.get_ylim()
            self.ax.set_xlim(x0 - dx_data, x1 - dx_data)
            self.ax.set_ylim(y0 - dy_data, y1 - dy_data)

            self._last_mouse_pos_px = (event.x, event.y)
            self.plot_function_with_points()

    def on_scroll(self, event):
        if event.inaxes:
            scale = 1.1 if event.step < 0 else 1 / 1.1  # Zoom in/out
            x0, x1 = self.ax.get_xlim()
            y0, y1 = self.ax.get_ylim()
            x, y = event.xdata, event.ydata

            new_xlim = [x - (x - x0) * scale, x + (x1 - x) * scale]
            new_ylim = [y - (y - y0) * scale, y + (y1 - y) * scale]

            self.ax.set_xlim(new_xlim)
            self.ax.set_ylim(new_ylim)
            self.plot_function_with_points()

    def get_styles(self):
        return """
        QWidget {
            font-family: Segoe UI, sans-serif;
            font-size: 14px;
            background-color: #f8f9fa;
            color: #2c3e50;
        }

        QTableWidget {
            background-color: white;
            border: 1px solid #ced4da;
            border-radius: 6px;
        }

        QHeaderView::section {
            background-color: #dee2e6;
            padding: 6px;
            border: none;
            font-weight: bold;
        }

        QTableWidget::item {
            padding: 6px;
        }

        QTableWidget::item:selected {
            background-color: #d0ebff;
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
            margin-bottom: 6px;
        }
        """

    def save_input_data(self):
        # Create saved_input directory if it doesn't exist
        saved_input_dir = "resources/saved_input"
        if not os.path.exists(saved_input_dir):
            os.makedirs(saved_input_dir)

        # Generate filename with current timestamp
        timestamp = datetime.now().strftime("%Y.%m.%d %H.%M.%S")
        filename = f"{timestamp}.json"
        filepath = os.path.join(saved_input_dir, filename)

        # Prepare data for saving
        save_data = {
            "function": self.input_data["expression"],
            "dimension": f"{1 if self.optimization_type == '1D' else len(self.input_data['start_point'])} dimension",
            "date": timestamp,
            "dimensionality": 1 if self.optimization_type == "1D" else len(self.input_data["start_point"]),
            "error": self.input_data["error"],
            "max_iterations": self.input_data["max_iterations"],
            "is_max": self.input_data["is_max"],
            "interval": {"x1": self.input_data["interval"]} if self.optimization_type == "1D" else None,
            "method_data": {}
        }

        # Dynamically import method classes
        try:
            from logic.methods import ONEDIMENSIONAL_METHODS_LIST, MULTIDIMENSIONAL_METHODS_LIST
        except ImportError:
            QtWidgets.QMessageBox.critical(
                self,
                "Помилка",
                "Не вдалося імпортувати класи методів оптимізації.",
                QtWidgets.QMessageBox.StandardButton.Ok
            )
            return

        # Build a mapping from method name to class instance
        method_classes = {}
        for cls in ONEDIMENSIONAL_METHODS_LIST if self.optimization_type == "1D" else MULTIDIMENSIONAL_METHODS_LIST:
            method_classes[cls().name] = cls

        # Add method-specific data using input_identifiers
        for method_name in self.results.keys():
            save_data["method_data"][method_name] = {}
            method_instance = method_classes[method_name]()
            for param in method_instance.input_identifiers:
                if param in self.input_data:
                    save_data["method_data"][method_name][param] = self.input_data[param]

        # Save to JSON file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=4, ensure_ascii=False)
            
            # Show success message
            QtWidgets.QMessageBox.information(
                self,
                "Успішне збереження",
                f"Вхідні дані збережено під іменем \"{filepath}\"",
                QtWidgets.QMessageBox.StandardButton.Ok
            )
        except Exception as e:
            # Show error message
            QtWidgets.QMessageBox.critical(
                self,
                "Помилка",
                f"Помилка при збереженні файлу: {str(e)}",
                QtWidgets.QMessageBox.StandardButton.Ok
            )

    def open_method_results(self, row, column):
        method_name = self.table.item(row, 0).text()
        method_result = {method_name: self.results[method_name]}
        viewer = IterationViewer(
            input_data=self.input_data,
            optimization_type=self.optimization_type,
            results=method_result
        )
        viewer.exec()

# Запуск
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ResultsWindow()
    window.exec()