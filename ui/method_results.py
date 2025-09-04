from PyQt6 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import sys
from logic.methods import ONEDIMENSIONAL_METHODS_LIST, MULTIDIMENSIONAL_METHODS_LIST
from logic.categories import OneDimensionalOptimizationCategory, MultiDimensionalOptimizationCategory
from controllers.input import input_identifier_to_widget
import json
import os
from datetime import datetime

class IterationViewer(QtWidgets.QDialog):
    def __init__(self, input_data=None, optimization_type="1D", results=None):
        super().__init__()
        self.setWindowTitle("Результати оптимізації")
        self.setWindowIcon(QtGui.QIcon('resources/images/icon.png'))
        self.setMinimumSize(1000, 600)
        self.setStyleSheet(self.get_styles())

        self.input_data = input_data
        self.optimization_type = optimization_type
        self.current_iteration = 0
        self.results = results if results is not None else {}
        self.method_names = list(self.results.keys())
        self.current_method_index = 0
        self._fixed_xlim = None  # Store initial interval for stationary plot
        self._fixed_ylim = None  # Store initial y-limits for stationary plot

        layout = QtWidgets.QHBoxLayout(self)

        # Graph
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)

        left_layout.addWidget(self.canvas, stretch=3)

        # Navigation buttons
        nav_layout = QtWidgets.QHBoxLayout()
        self.prev_button = QtWidgets.QPushButton("← Попередня ітерація")
        self.next_button = QtWidgets.QPushButton("Наступна ітерація →")
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)
        left_layout.addLayout(nav_layout)
        
        layout.addWidget(left_panel, stretch=3)

        # Right panel
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)

        # Method selector (hidden if only one method)
        method_layout = QtWidgets.QHBoxLayout()
        self.method_label = QtWidgets.QLabel("Метод:")
        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.addItems(self.method_names)
        if len(self.method_names) <= 1:
            self.method_combo.setEnabled(False)
            self.method_combo.setVisible(False)
            self.method_label.setVisible(False)
        method_layout.addWidget(self.method_label)
        method_layout.addWidget(self.method_combo)
        right_layout.addLayout(method_layout)

        # Input data display
        self.input_label = QtWidgets.QLabel()
        self.input_label.setWordWrap(True)
        self.input_label.setObjectName("InputLabel")
        right_layout.addWidget(self.input_label, 4)

        # Iterations display
        self.text_label = QtWidgets.QLabel()
        self.text_label.setWordWrap(True)
        self.text_label.setObjectName("IterationLabel")
        right_layout.addWidget(self.text_label, 6)

        self.export_button = QtWidgets.QPushButton("Експортувати результати")
        right_layout.addWidget(self.export_button, alignment=QtCore.Qt.AlignmentFlag.AlignRight)

        layout.addWidget(right_panel, stretch=2)

        # Connect signals
        self.prev_button.clicked.connect(self.prev_iteration)
        self.next_button.clicked.connect(self.next_iteration)
        self.method_combo.currentIndexChanged.connect(self.on_method_changed)
        self.export_button.clicked.connect(self.export_current_method_result)

        self.update_view()

    def on_method_changed(self, index):
        self.current_method_index = index
        self.current_iteration = 0
        self.update_view()

    def update_view(self):
        if not self.method_names:
            return
        method_name = self.method_names[self.current_method_index]
        result = self.results.get(method_name)
        if result:
            self.plot_current_iteration(result)
            self.update_text(result, method_name)
            iterations = result.get('iterations', {})
            self.prev_button.setEnabled(self.current_iteration > 0)
            self.next_button.setEnabled(self.current_iteration < len(iterations) - 1)
        else:
            self.ax.clear()
            self.ax.text(0.5, 0.5, "Немає даних для відображення", 
                        transform=self.ax.transAxes, ha='center', va='center')
            self.canvas.draw()
            self.text_label.setText("Немає даних для відображення")

    def compute_function(self, x):
        expr = self.input_data.get('expression', '0').replace('^', '**')
        try:
            # Allow numpy and common math functions
            return eval(expr, {"x": x, "np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "abs": np.abs})
        except Exception as e:
            print(f"Error evaluating function: {e}")
            return np.zeros_like(x)

    def plot_current_iteration(self, result):
        self.ax.clear()
        iterations = result.get('iterations', {})
        iter_keys = list(iterations.keys())
        if not iter_keys or self.current_iteration >= len(iter_keys):
            return
        # Use the initial interval for all plots
        if self._fixed_xlim is None:
            first_iter = iterations[iter_keys[0]]
            a = first_iter.get('a', 0)
            b = first_iter.get('b', 0)
            self._fixed_xlim = (a, b)
            x_full = np.linspace(a, b, 1000)
            y_full = self.compute_function(x_full)
            y_min, y_max = np.min(y_full), np.max(y_full)
            y_pad = (y_max - y_min) * 0.1 if y_max > y_min else 1
            self._fixed_ylim = (y_min - y_pad, y_max + y_pad)
        a, b = self._fixed_xlim
        x = np.linspace(a, b, 1000)
        y = self.compute_function(x)
        self.ax.plot(x, y, color="#2c3e50")
        # Get the method's iteration_result_key
        method_name = self.method_names[self.current_method_index]
        method_cls = None
        if self.optimization_type == "1D":
            method_list = ONEDIMENSIONAL_METHODS_LIST
        else:
            method_list = MULTIDIMENSIONAL_METHODS_LIST
        for cls in method_list:
            if cls().name == method_name:
                method_cls = cls
                break
        iteration_result_key = getattr(method_cls(), "iteration_result_key", None) if method_cls else None
        # Collect all possible points for 1D methods dynamically
        xs, ys = [], []
        all_points = []
        for k in iter_keys:
            iter_data = iterations[k]
            for key in iter_data.keys():
                # Accept keys that are likely to be points: start with 'x', 'c', 'd', or match iteration_result_key
                if (str(key).startswith(('x', 'c', 'd')) or key == iteration_result_key):
                    fkey = f"f({key})" if key != 'x' else 'f(x)'
                    if key in iter_data and fkey in iter_data:
                        xs.append(iter_data[key])
                        ys.append(iter_data[fkey])
                        all_points.append((iter_data[key], iter_data[fkey], k))
        # Plot all previous/future iterations as light blue
        if xs and ys:
            self.ax.scatter(xs, ys, color='#90caf9', s=80, zorder=3, alpha=0.7)
        # Highlight only the current iteration's main result key
        current_iter = iterations[iter_keys[self.current_iteration]]
        if iteration_result_key:
            key = iteration_result_key
            fkey = f"f({key})" if key != 'x' else 'f(x)'
            if key in current_iter and fkey in current_iter:
                val = (current_iter[key], current_iter[fkey])
                count = sum(1 for pt in all_points if pt[0] == val[0] and pt[1] == val[1])
                if count < len(iter_keys):
                    self.ax.scatter([current_iter[key]], [current_iter[fkey]], color='#1976d2', s=140, zorder=4, alpha=1.0)
        self.ax.set_xlim(*self._fixed_xlim)
        self.ax.set_ylim(*self._fixed_ylim)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("f(x)")
        self.ax.grid(True)
        self.canvas.draw()

    def update_text(self, result, method_name):
        # Get the method class and its input identifiers
        if self.optimization_type == "1D":
            method_list = ONEDIMENSIONAL_METHODS_LIST
            category = OneDimensionalOptimizationCategory()
        else:
            method_list = MULTIDIMENSIONAL_METHODS_LIST
            category = MultiDimensionalOptimizationCategory()
        method_cls = next((cls for cls in method_list if cls().name == method_name), None)
        if method_cls:
            method_instance = method_cls()
            identifiers = set(getattr(method_instance, "input_identifiers", []))
        else:
            identifiers = set()
        # Always include category common identifiers
        identifiers.update(getattr(category, "input_identifiers", []))
        # Build input data display using user-friendly labels
        input_text = "<b>Вхідні дані</b><br><br>"
        for key in identifiers:
            if key == "function":
                try:
                    label = input_identifier_to_widget(key)["label"]
                except Exception:
                    label = key
                value = self.input_data.get("expression", "")
                input_text += f"{label}: {value}<br>"
            elif key in self.input_data:
                try:
                    label = input_identifier_to_widget(key)["label"]
                except Exception:
                    label = key
                value = self.input_data[key]
                input_text += f"{label}: {value}<br>"
        self.input_label.setText(input_text)
        # Iteration info
        iterations = result.get('iterations', {})
        iter_keys = list(iterations.keys())
        if not iter_keys or self.current_iteration >= len(iter_keys):
            self.text_label.setText("")
            return
        # Use method-specific description if available
        if method_cls and hasattr(method_cls, "get_iteration_descriptions"):
            method_instance = method_cls()
            method_instance.result = result
            descriptions = method_instance.get_iteration_descriptions()
            if 0 <= self.current_iteration < len(descriptions):
                iter_text = f"<b>Ітерація {self.current_iteration + 1}</b><br><br>" + descriptions[self.current_iteration].replace("\n", "<br>")
                self.text_label.setText(iter_text)
                return
        # Fallback: generic display
        current_iter = iterations[iter_keys[self.current_iteration]]
        iter_text = f"<b>Ітерація {self.current_iteration + 1}</b><br><br>"
        for key, value in current_iter.items():
            iter_text += f"{key}: {value}<br>"
        self.text_label.setText(iter_text)

    def prev_iteration(self):
        if self.current_iteration > 0:
            self.current_iteration -= 1
            self.update_view()

    def next_iteration(self):
        method_name = self.method_names[self.current_method_index]
        result = self.results.get(method_name)
        iterations = result.get('iterations', {})
        if self.current_iteration < len(iterations) - 1:
            self.current_iteration += 1
            self.update_view()

    def export_current_method_result(self):
        # Ensure output directory exists
        output_dir = os.path.join("resources", "output")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Get current method name and result
        method_name = self.method_names[self.current_method_index]
        result = self.results.get(method_name, {})
        # Prepare filename
        timestamp = datetime.now().strftime("%Y.%m.%d_%H-%M-%S")
        filename = f"{method_name}_{timestamp}.json"
        filename = filename.replace(' ', '_')  # Remove spaces from method name
        filepath = os.path.join(output_dir, filename)
        # Save result as JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        # Optional: show a message box
        QtWidgets.QMessageBox.information(self, "Експорт завершено", f"Результати методу '{method_name}' експортовано у файл:\n{filepath}")

    def get_styles(self):
        return """
        /* Загальні налаштування */
        QWidget {
            font-family: 'Segoe UI', sans-serif;
            font-size: 15px;
            background-color: #f4f6f8;
            color: #2c3e50;
        }

        /* Таблиця */
        QTableWidget {
            background-color: #ffffff;
            border: 1px solid #d1dce5;
            border-radius: 10px;
            padding: 4px;
            gridline-color: #e0e6ed;
            alternate-background-color: #f9fbfd;
        }

        QHeaderView::section {
            background-color: #f0f4f8;
            padding: 8px 12px;
            border: none;
            font-weight: 600;
            font-size: 14px;
            color: #34495e;
        }

        QTableWidget::item {
            padding: 8px 10px;
        }

        QTableWidget::item:selected {
            background-color: #d0ebff;
            color: #1b2a41;
        }

        /* Кнопки */
        QPushButton {
            background-color: #3b82f6;
            color: white;
            padding: 10px 20px;
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.3s ease;
        }

        QPushButton:hover {
            background-color: #60a5fa;
        }

        QPushButton:pressed {
            background-color: #2563eb;
        }

        QPushButton:disabled {
            background-color: #e0e0e0;
            color: #888888;
            border: none;
        }

        /* Текстові блоки */
        QLabel#InputLabel {
            background-color: #a8e2ff;
            border: 1px solid #90cdf4;
            border-radius: 10px;
            padding: 14px;
            margin-bottom: 16px;
            font-size: 14.5px;
            color: #0c4a6e;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }

        QLabel#IterationLabel {
            background-color: #e0f2fe;
            border: 1px solid #90cdf4;
            border-radius: 10px;
            padding: 14px;
            font-size: 15px;
            color: #614700;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            margin-bottom: 16px;
        }

        /* Глобальні заголовки або інші QLabel */
        QLabel {
            margin-bottom: 6px;
        }

        /* Спеціально для діалогових кнопок */
        QDialog QPushButton {
            margin-top: 12px;
        }

        /* Додаткові відступи для панелей */
        QVBoxLayout, QHBoxLayout {
            spacing: 10px;
        }
        """


# Тестовий запуск
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = IterationViewer()
    window.exec()