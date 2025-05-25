import numpy, sympy
from PyQt6 import QtWidgets, QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from logic.categories import OneDimensionalOptimizationCategory, MultiDimensionalOptimizationCategory
from logic.methods import ONEDIMENSIONAL_METHODS_LIST, MULTIDIMENSIONAL_METHODS_LIST
from controllers.input import *

def create_scroll_area():
    area = QtWidgets.QScrollArea()
    area.setWidgetResizable(True)
    content = QtWidgets.QWidget()
    area.setWidget(content)
    return area

class InputPanel(QtWidgets.QGroupBox):
    def __init__(self, title="Вхідні дані"):
        super().__init__(title)
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)
        self.widgets = {}

    def clear(self):
        while self.layout.count():
            child = self.layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.widgets.clear()

    def add_input(self, label_text, widget):
        # label = QtWidgets.QLabel(label_text)
        # self.layout.addWidget(label)
        self.layout.addWidget(widget)
        self.widgets[label_text] = widget

    def get_value(self, label_text):
        widget = self.widgets.get(label_text)
        if isinstance(widget, QtWidgets.QLineEdit):
            return widget.text()
        elif isinstance(widget, QtWidgets.QComboBox):
            return widget.currentText()
        return None

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Оптимізація")
        self.setGeometry(0, 0, 1500, 800)

        self._dragging = False
        self._last_mouse_pos_px = None
        self.optimization_type = "1D"
        self.dimensionality = 1

        self._widgets = {}
        self._layouts = {}
        self._input_widgets = {}
        self._methods_checkboxes = {}

        self.setup_ui()
        self.set_optimization_type("1D")
        self.update_plot()
        self.update_methods_widget()



    def setup_ui(self):
        self._widgets["central_widget"] = QtWidgets.QWidget()
        self.setCentralWidget(self._widgets["central_widget"])
        self._layouts["main_layout"] = QtWidgets.QHBoxLayout(self._widgets["central_widget"])

        self.setup_left_area()
        self.setup_right_area()

    def setup_left_area(self):
        self._widgets["left_scroll_area"] = create_scroll_area()
        self._layouts["left_layout"] = QtWidgets.QVBoxLayout()
        self._widgets["left_scroll_area"].widget().setLayout(self._layouts["left_layout"])
        self._layouts["main_layout"].addWidget(self._widgets["left_scroll_area"], stretch=1)

        self.setup_optimization_type_selector()
        self.setup_graph_widget()
        self.setup_input_widget()

    def setup_right_area(self):
        self._widgets["right_scroll_area"] = create_scroll_area()
        self._layouts["right_layout"] = QtWidgets.QVBoxLayout()
        self._widgets["right_scroll_area"].widget().setLayout(self._layouts["right_layout"])
        self._layouts["main_layout"].addWidget(self._widgets["right_scroll_area"], stretch=1)

        self.setup_methods_widget()
        self.setup_buttons_widget()

    def setup_optimization_type_selector(self):
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(QtWidgets.QLabel("Тип оптимізації:"))

        self._widgets["1D_button"] = QtWidgets.QPushButton("Одновимірна")
        self._widgets["ND_button"] = QtWidgets.QPushButton("Багатовимірна")

        for key in ["1D_button", "ND_button"]:
            self._widgets[key].setCheckable(True)
            layout.addWidget(self._widgets[key])

        self._widgets["1D_button"].clicked.connect(lambda: self.set_optimization_type("1D"))
        self._widgets["ND_button"].clicked.connect(lambda: self.set_optimization_type("ND"))

        self._layouts["left_layout"].addLayout(layout)

    def setup_graph_widget(self):
        self._widgets["graph_area"] = create_scroll_area()
        self._layouts["graph_layout"] = QtWidgets.QVBoxLayout()
        self._widgets["graph_area"].widget().setLayout(self._layouts["graph_layout"])
        self._layouts["left_layout"].addWidget(self._widgets["graph_area"], stretch=3)

        self._widgets["coord_label"] = QtWidgets.QLabel("x: -, y: -")
        self._widgets["coord_label"].setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignTop)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        canvas_frame = QtWidgets.QFrame()
        layout = QtWidgets.QVBoxLayout(canvas_frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._widgets["coord_label"])
        layout.addWidget(self.canvas)

        self._widgets["canvas_container"] = QtWidgets.QStackedLayout()
        canvas_widget = QtWidgets.QWidget()
        canvas_widget.setLayout(self._widgets["canvas_container"])
        self._widgets["canvas_container"].addWidget(canvas_frame)
        self._layouts["graph_layout"].addWidget(canvas_widget)

        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.canvas.mpl_connect("motion_notify_event", self.update_cursor_position)

    def setup_input_widget(self):
        self._widgets["input_data_area"] = create_scroll_area()
        self._layouts["left_layout"].addWidget(self._widgets["input_data_area"], stretch=2)

        self._layouts["input_data_layout"] = QtWidgets.QVBoxLayout()
        self._layouts["input_data_layout"].setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self._widgets["input_data_area"].widget().setLayout(self._layouts["input_data_layout"])

        self.input_panel = InputPanel()
        self._layouts["input_data_layout"].addWidget(self.input_panel)

    def setup_methods_widget(self):
        self._widgets["methods_list_area"] = create_scroll_area()
        self._widgets["methods_list_content"] = self._widgets["methods_list_area"].widget()
        self._layouts["methods_list_layout"] = QtWidgets.QVBoxLayout(self._widgets["methods_list_content"])
        self._layouts["methods_list_layout"].setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self._layouts["right_layout"].addWidget(self._widgets["methods_list_area"], stretch=8)

    def setup_buttons_widget(self):
        self._widgets["start_buttons_area"] = create_scroll_area()
        self._layouts["start_buttons_layout"] = QtWidgets.QVBoxLayout()
        self._widgets["start_buttons_area"].widget().setLayout(self._layouts["start_buttons_layout"])

        layout = QtWidgets.QHBoxLayout()
        self._widgets["calculate_button"] = QtWidgets.QPushButton("Обчислити")
        self._widgets["load_button"] = QtWidgets.QPushButton("Завантажити вхідні дані")
        layout.addWidget(self._widgets["calculate_button"])
        layout.addWidget(self._widgets["load_button"])

        self._layouts["start_buttons_layout"].addLayout(layout)
        self._layouts["right_layout"].addWidget(self._widgets["start_buttons_area"], stretch=1)



    def set_optimization_type(self, opt_type):
        self.optimization_type = opt_type

        self._widgets["1D_button"].setChecked(opt_type == "1D")
        self._widgets["ND_button"].setChecked(opt_type == "ND")

        self._widgets["1D_button"].setStyleSheet("background-color: lightblue;" if opt_type == "1D" else "")
        self._widgets["ND_button"].setStyleSheet("background-color: lightblue;" if opt_type == "ND" else "")

        self.update_methods_widget()
        self.update_input_parameters()

    def update_methods_widget(self):
        for i in reversed(range(self._widgets["methods_list_content"].layout().count())):
            widget = self._widgets["methods_list_content"].layout().itemAt(i).widget()
            if widget:
                widget.setParent(None)

        method_list = ONEDIMENSIONAL_METHODS_LIST if self.optimization_type == "1D" else MULTIDIMENSIONAL_METHODS_LIST

        self.select_all_checkbox = QtWidgets.QCheckBox("Вибрати всі")
        self.select_all_checkbox.stateChanged.connect(self.on_select_all_methods)
        self._widgets["methods_list_content"].layout().addWidget(self.select_all_checkbox)

        self._methods_checkboxes = {}  # додано: збереження відповідностей

        for method in method_list:
            checkbox = QtWidgets.QCheckBox(method().name)
            checkbox.stateChanged.connect(self.on_method_selection_changed)
            self._widgets["methods_list_content"].layout().addWidget(checkbox)
            self._methods_checkboxes[method().name] = checkbox

    def update_plot(self):
        self.update_input_data()

        x_sym = sympy.Symbol('x')

        if self.input_data["Спільні вхідні дані"]["function"] != None or (self.input_data["Спільні вхідні дані"]["dimensionality"] != None if "dimensionality" in self.input_data["Спільні вхідні дані"] else True):
            self.ax.clear()
            self.ax.grid(True)
            self.ax.axhline(0, color='black')
            self.ax.axvline(0, color='black')
            self.ax.set_aspect('equal', adjustable='datalim')
            self.canvas.draw()
            return

        try:
            try:
                x0, x1 = self.ax.get_xlim()
                if not numpy.isfinite([x0, x1]).all():
                    raise ValueError
            except Exception:
                x0, x1 = -10, 10

            x0, x1 = x0 - 5, x1 + 5
            x_vals = numpy.linspace(x0, x1, 1000)
            expr = sympy.sympify(self.input_widget_groups["Спільні вхідні дані"].get_input("function").text(), evaluate=False)
            func = sympy.lambdify(x_sym, expr, modules=["numpy"])
            y_vals = func(x_vals)

            if hasattr(self, 'plot_line') and self.plot_line in self.ax.lines:
                self.plot_line.set_data(x_vals, y_vals)
            else:
                self.ax.clear()
                self.plot_line, = self.ax.plot(x_vals, y_vals)
                self.ax.grid(True)
                self.ax.axhline(0, color='black')
                self.ax.axvline(0, color='black')
                self.ax.set_aspect('equal', adjustable='datalim')

            try:
                self.input_widget_groups["Спільні вхідні дані"].get_input("function").line_edit.setStyleSheet("")
            except RuntimeError:
                pass

        except Exception as e:
            self.ax.clear()
            self.ax.text(0.5, 0.5, f"Error: {e}", transform=self.ax.transAxes,
                        ha='center', va='center', color='red')
            try:
                self.input_widget_groups["Спільні вхідні дані"].get_input("function").line_edit.setStyleSheet("border: 2px solid red;")
            except RuntimeError:
                pass

        self.canvas.draw()
        
    def update_cursor_position(self, event):
        if event.inaxes:
            self._widgets["coord_label"].setText(f"x: {event.xdata:.2f}, y: {event.ydata:.2f}")
        else:
            self._widgets["coord_label"].setText("")

    def update_input_parameters(self):
        input_identifiers = {"Спільні вхідні дані": (OneDimensionalOptimizationCategory if self.optimization_type == "1D" else MultiDimensionalOptimizationCategory)().input_identifiers}

        # Назви методів, які обрано користувачем
        selected_methods_names = [
            name for name, cb in self._methods_checkboxes.items()
            if cb.isChecked()
        ]

        method_list = ONEDIMENSIONAL_METHODS_LIST if self.optimization_type == "1D" else MULTIDIMENSIONAL_METHODS_LIST

        for method in method_list:
            if method().name in selected_methods_names:
                if len(method().input_identifiers) != 0:
                    input_identifiers[method().name] = method().input_identifiers

        self.input_panel.clear()
        self.input_widget_groups = build_input_groups(input_identifiers, self.optimization_type, self.dimensionality, self.input_panel)

        for title, widget in self.input_widget_groups.items():
            self.input_panel.add_input(title, widget)

        self.input_widget_groups["Спільні вхідні дані"].get_input("function").line_edit.textChanged.connect(self.on_input_changed)
        if self.dimensionality == "ND":
            self.input_widget_groups["Спільні вхідні дані"].get_input("dimensionality").line_edit.textChanged.connect(self.on_input_changed)

    def update_input_data(self):
        """Збирає всі введені користувачем дані з віджетів і зберігає у self.data"""
        self.input_data = {}

        for name, group in self.input_widget_groups.items():
            self.input_data[name] = {}
            for identifier, input in group.inputs.items():
                self.input_data[name][identifier] = input.get_data()

        return self.input_data



    def on_press(self, event):
        if event.button == 1 and event.inaxes:
            self._dragging = True
            self._last_mouse_pos_px = (event.x, event.y)

    def on_release(self, event):
        self._dragging = False
        self._last_mouse_pos_px = None

    def on_motion(self, event):
        if self._dragging and event.inaxes and self._last_mouse_pos_px:
            dx_px, dy_px = event.x - self._last_mouse_pos_px[0], event.y - self._last_mouse_pos_px[1]
            inv = self.ax.transData.inverted()
            dx_data, dy_data = inv.transform((dx_px, dy_px)) - inv.transform((0, 0))

            x0, x1 = self.ax.get_xlim()
            y0, y1 = self.ax.get_ylim()
            self.ax.set_xlim(x0 - dx_data, x1 - dx_data)
            self.ax.set_ylim(y0 - dy_data, y1 - dy_data)

            self._last_mouse_pos_px = (event.x, event.y)

            self.update_plot()
            self.canvas.draw_idle()

    def on_scroll(self, event):
        if event.inaxes:
            scale = 1 / 1.1 if event.step > 0 else 1.1
            x0, x1 = self.ax.get_xlim()
            y0, y1 = self.ax.get_ylim()
            x, y = event.xdata, event.ydata

            new_xlim = [x - (x - x0) * scale, x + (x1 - x) * scale]
            new_ylim = [y - (y - y0) * scale, y + (y1 - y) * scale]

            self.ax.set_xlim(new_xlim)
            self.ax.set_ylim(new_ylim)

            self.update_plot()
            self.canvas.draw_idle()

    def on_select_all_methods(self, state):
        for checkbox in self._methods_checkboxes.values():
            checkbox.setChecked(state)
        self.update_input_parameters()

    def on_method_selection_changed(self):
        all_checked = all(cb.isChecked() for cb in self._methods_checkboxes.values())
        self.select_all_checkbox.blockSignals(True)
        self.select_all_checkbox.setChecked(all_checked)
        self.select_all_checkbox.blockSignals(False)
        self.update_input_parameters()
    
    def on_input_changed(self):
        self.update_plot()