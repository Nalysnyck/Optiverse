import numpy, sympy
from PyQt6 import QtWidgets, QtCore, QtGui
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg \
    as FigureCanvas

from logic.categories import OneDimensionalOptimizationCategory,\
     MultiDimensionalOptimizationCategory
from logic.methods import ONEDIMENSIONAL_METHODS_LIST, \
    MULTIDIMENSIONAL_METHODS_LIST
from controllers.input import *
from ui.load_data import SavedDataWindow
from controllers.function import create_function, get_derivative, get_gradient, get_hessian

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
        self.layout.addWidget(widget)
        self.widgets[label_text] = widget

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Optiverse")
        self.setMinimumSize(1200, 800)
        self.setWindowIcon(QtGui.QIcon('resources/images/icon.png'))

        self.optimization_type = "1D"
        self.dimensionality = 1
        self.input_widget_groups = {}
        self.input_data = {}
        self._drag_start = None
        self._dragging = False
        self._last_mouse_pos_px = None
        self._last_dx = 0  # For smoothing
        self._last_dy = 0  # For smoothing

        self._widgets = {}
        self._layouts = {}
        self._input_widgets = {}
        self._methods_checkboxes = {}

        self.setup_ui()
        self.set_optimization_type("1D")
        self.update_plot()
        self.update_methods_widget()
        self.apply_styles()

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
        self._layouts["main_layout"].addWidget(self._widgets["left_scroll_area"], stretch=7)

        self.setup_optimization_type_selector()
        self.setup_graph_widget()
        self.setup_input_widget()

    def setup_right_area(self):
        self._widgets["right_scroll_area"] = create_scroll_area()
        self._layouts["right_layout"] = QtWidgets.QVBoxLayout()
        self._widgets["right_scroll_area"].widget().setLayout(self._layouts["right_layout"])
        self._layouts["main_layout"].addWidget(self._widgets["right_scroll_area"], stretch=4)

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
        self._widgets["coord_label"].setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | \
                                                  QtCore.Qt.AlignmentFlag.AlignTop)

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
        self._layouts["methods_list_layout"] = QtWidgets.QVBoxLayout(\
            self._widgets["methods_list_content"])
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

        self._widgets["calculate_button"].clicked.connect(self.validate_input)
        self._widgets["load_button"].clicked.connect(self.show_load_data_window)

    def validate_input(self):
        # First validate all inputs
        all_valid = True
        for group in self.input_widget_groups.values():
            for input in group.inputs.values():
                if not input.is_valid():
                    input.set_invalid()
                    all_valid = False

        if not all_valid:
            return

        # Get all input data
        raw_input_data = self.update_input_data()
        common = raw_input_data["Спільні вхідні дані"]

        # Get selected methods
        selected_methods = []
        method_list = ONEDIMENSIONAL_METHODS_LIST if self.optimization_type == "1D" else MULTIDIMENSIONAL_METHODS_LIST
        for method in method_list:
            if method().name in self._methods_checkboxes and self._methods_checkboxes[method().name].isChecked():
                selected_methods.append(method)

        if not selected_methods:
            QtWidgets.QMessageBox.warning(self, "Попередження", "Будь ласка, виберіть хоча б один метод оптимізації.")
            return

        input_data = {}
        input_data["expression"] = common["function"]
        input_data["error"] = float(common.get("error", 0.001))
        input_data["is_max"] = bool(common.get("is_max", False))
        input_data["max_iterations"] = int(common.get("max_iterations", 500))

        if self.optimization_type == "1D":
            # Interval parsing
            interval = common.get("interval", "")
            if isinstance(interval, str):
                try:
                    interval = interval.strip("[](){} ").split(",")
                    interval = [float(x.strip()) for x in interval]
                except Exception as e:
                    QtWidgets.QMessageBox.warning(self, "Помилка", f"Неправильний формат інтервалу: {str(e)}")
                    return
            input_data["interval"] = interval['x1']
            input_data["start_point"] = interval['x1'][0]
            input_data["function"] = create_function(common["function"])
            input_data["derivative1"] = create_function(get_derivative(common["function"]))
            input_data["derivative2"] = create_function(get_derivative(get_derivative(common["function"])))
        else:
            # ND case
            dim = int(common.get("dimensionality", 2))
            input_data["dimensionality"] = dim
            start_point = common.get("start_point", [0]*dim)
            if isinstance(start_point, str):
                try:
                    start_point = start_point.strip("[](){} ").split(",")
                    start_point = [float(x.strip()) for x in start_point]
                except Exception as e:
                    QtWidgets.QMessageBox.warning(self, "Помилка", f"Неправильний формат початкової точки: {str(e)}")
                    return
            input_data["start_point"] = start_point
            variables = [f"x{i+1}" for i in range(dim)]
            input_data["function"] = create_function(common["function"], variables)
            input_data["gradient"] = get_gradient(common["function"], variables)
            input_data["hessian"] = get_hessian(common["function"], variables)
            input_data["bounds"] = [(-10, 10)] * dim
            input_data["expression"] = common["function"]
            input_data["variables"] = variables

        # Add method-specific params
        for method in selected_methods:
            method_name = method().name
            if method_name in raw_input_data:
                input_data.update(raw_input_data[method_name])

        # Compute results for all selected methods
        results = {}
        for method_class in selected_methods:
            method = method_class()
            method.run(input_data)
            results[method.name] = method.result

        # Create and show results window
        from ui.results import ResultsWindow
        results_window = ResultsWindow(
            input_data=input_data,
            optimization_type=self.optimization_type,
            results=results
        )
        results_window.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        results_window.exec()

    def set_optimization_type(self, opt_type):
        self.optimization_type = opt_type
        # Reset dimensionality when switching modes
        self.dimensionality = 1 if opt_type == "1D" else 2

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

        method_list = ONEDIMENSIONAL_METHODS_LIST if self.optimization_type == "1D" \
            else MULTIDIMENSIONAL_METHODS_LIST

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

        if not self.input_data["Спільні вхідні дані"]["function"] or (self.input_data["Спільні вхідні дані"]["dimensionality"] == None if "dimensionality" in self.input_data["Спільні вхідні дані"] else False):
            self.figure.clear()
            self.ax = self.figure.add_subplot(111)
            self.ax.grid(True)
            self.ax.axhline(0, color='black')
            self.ax.axvline(0, color='black')
            self.ax.set_aspect('equal', adjustable='datalim')
            self.canvas.draw()
            return

        try:
            # Get current visible area
            try:
                x0, x1 = self.ax.get_xlim()
                if not numpy.isfinite([x0, x1]).all():
                    raise ValueError
            except Exception:
                x0, x1 = -10, 10

            # Add some padding to ensure the function is fully visible
            padding = (x1 - x0) * 0.05  # Reduced padding from 0.1 to 0.05
            x0 -= padding
            x1 += padding
            
            # Add limits to prevent excessive ranges
            max_range = 1e6
            if abs(x1 - x0) > max_range:
                center = (x0 + x1) / 2
                x0 = center - max_range/2
                x1 = center + max_range/2
            
            # Check if we're dealing with a 2D function
            if self.optimization_type == "ND" and self.dimensionality == 2:
                # Create 3D plot
                self.figure.clear()
                self.ax = self.figure.add_subplot(111, projection='3d')
                
                # Create meshgrid for 3D surface with fewer points for better performance
                x = numpy.linspace(x0, x1, 50)
                y = numpy.linspace(x0, x1, 50)
                X, Y = numpy.meshgrid(x, y)
                
                # Create symbols for both variables
                x_sym, y_sym = sympy.symbols('x1 x2')
                expr = sympy.sympify(self.input_data["Спільні вхідні дані"]["function"], evaluate=False)
                func = sympy.lambdify((x_sym, y_sym), expr, modules=["numpy"])
                
                # Compute Z values
                Z = func(X, Y)
                
                # Plot 3D surface with reduced quality for better performance
                surf = self.ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                                         rcount=50, ccount=50,
                                         antialiased=False)
                self.figure.colorbar(surf, ax=self.ax, shrink=0.5, aspect=5)
                
                self.ax.set_xlabel('x1')
                self.ax.set_ylabel('x2')
                self.ax.set_zlabel('f(x1, x2)')
                
            else:
                # Reset to 2D plot
                self.figure.clear()
                self.ax = self.figure.add_subplot(111)
                
                # Create more points for smoother 2D plot
                x_vals = numpy.linspace(x0-5, x1+5, 2000)
                x_sym = sympy.Symbol('x')
                expr = sympy.sympify(self.input_data["Спільні вхідні дані"]["function"], evaluate=False)
                func = sympy.lambdify(x_sym, expr, modules=["numpy"])
                y_vals = func(x_vals)

                self.plot_line, = self.ax.plot(x_vals, y_vals)
                self.ax.grid(True)
                self.ax.axhline(0, color='black')
                self.ax.axvline(0, color='black')
                self.ax.set_aspect('equal', adjustable='datalim')

            try:
                self.input_widget_groups["Спільні вхідні дані"].get_input("function").set_valid()
            except RuntimeError:
                pass
        
        except Exception as e:
            self.figure.clear()
            self.ax = self.figure.add_subplot(111)
            self.ax.text(0.5, 0.5, f"Error: {e}", transform=self.ax.transAxes,
                        ha='center', va='center', color='red')
            try:
                self.input_widget_groups["Спільні вхідні дані"].get_input("function").set_invalid()
            except RuntimeError:
                pass

        self.canvas.draw()

    def update_cursor_position(self, event):
        if event.inaxes:
            if self.optimization_type == "ND" and self.dimensionality == 2:
                # For 3D plots, we need to project the mouse position onto the surface
                x, y = event.xdata, event.ydata
                if x is not None and y is not None:
                    try:
                        x_sym, y_sym = sympy.symbols('x1 x2')
                        expr = sympy.sympify(self.input_data["Спільні вхідні дані"]["function"], evaluate=False)
                        # Create a substitution dictionary
                        subs = {x_sym: float(x), y_sym: float(y)}
                        # Evaluate the expression with the substitutions
                        z = float(expr.subs(subs).evalf())
                        self._widgets["coord_label"].setText(f"x1: {float(x):.2f}, x2: {float(y):.2f}, f(x1,x2): {z:.2f}")
                    except Exception:
                        self._widgets["coord_label"].setText("")
            else:
                self._widgets["coord_label"].setText(f"x: {float(event.xdata):.2f}, y: {float(event.ydata):.2f}")
        else:
            self._widgets["coord_label"].setText("")

    def update_input_parameters(self):
        input_identifiers = {"Спільні вхідні дані": (OneDimensionalOptimizationCategory \
        if self.optimization_type == "1D" else MultiDimensionalOptimizationCategory)().input_identifiers}

        # Назви методів, які обрано користувачем
        selected_methods_names = [
            name for name, cb in self._methods_checkboxes.items()
            if cb.isChecked()
        ]

        method_list = ONEDIMENSIONAL_METHODS_LIST if self.optimization_type == "1D" \
            else MULTIDIMENSIONAL_METHODS_LIST

        for method in method_list:
            if method().name in selected_methods_names:
                if len(method().input_identifiers) != 0:
                    input_identifiers[method().name] = method().input_identifiers

        self.input_panel.clear()
        self.input_widget_groups = build_input_groups(input_identifiers, self.optimization_type, \
                                                      self.dimensionality, self.input_panel)

        for title, widget in self.input_widget_groups.items():
            self.input_panel.add_input(title, widget)

        self.input_widget_groups["Спільні вхідні дані"].get_input("function").\
            line_edit.textChanged.connect(self.on_input_changed)
        if self.optimization_type == "ND":
            self.input_widget_groups["Спільні вхідні дані"].get_input("dimensionality").\
                line_edit.textChanged.connect(self.on_input_changed)

    def update_input_data(self):
        """Збирає всі введені користувачем дані з віджетів і зберігає у self.data"""
        self.input_data = {}

        for name, group in self.input_widget_groups.items():
            self.input_data[name] = {}
            for identifier, input in group.inputs.items():
                self.input_data[name][identifier] = input.get_data() if input.is_valid() else None

        return self.input_data

    def on_press(self, event):
        if hasattr(self.ax, 'get_zlim'):
            # 3D plot: use previous logic
            if event.inaxes != self.ax:
                return
            self._drag_start = (event.x, event.y)
            self.canvas.draw_idle()
        else:
            # 2D plot: store initial position in data coordinates
            if event.button == 1 and event.inaxes:
                self._dragging = True
                self._last_mouse_pos_px = (event.xdata, event.ydata)

    def on_release(self, event):
        if hasattr(self.ax, 'get_zlim'):
            self._drag_start = None
            self.canvas.draw_idle()
        else:
            self._dragging = False
            self._last_mouse_pos_px = None
            self._last_dx = 0  # Reset smoothing
            self._last_dy = 0  # Reset smoothing

    def on_motion(self, event):
        if hasattr(self.ax, 'get_zlim'):
            # 3D plot: use previous logic
            if event.inaxes != self.ax or self._drag_start is None:
                return
            elev, azim = self.ax.elev, self.ax.azim
            dx = event.x - self._drag_start[0]
            dy = event.y - self._drag_start[1]
            self.ax.view_init(elev=elev - dy/2, azim=azim + dx/2)
            self._drag_start = (event.x, event.y)
            self.canvas.draw_idle()
        else:
            # 2D plot: use proper coordinate transformation
            if self._dragging and event.inaxes and self._last_mouse_pos_px:
                # Get the current view limits
                x0, x1 = self.ax.get_xlim()
                y0, y1 = self.ax.get_ylim()
                
                # Get the current mouse position in data coordinates
                current_x = event.xdata
                current_y = event.ydata
                
                # Get the last mouse position in data coordinates
                last_x = self._last_mouse_pos_px[0]
                last_y = self._last_mouse_pos_px[1]
                
                # Calculate the movement in data coordinates
                dx = current_x - last_x
                dy = current_y - last_y
                
                # Get the current ranges
                x_range = x1 - x0
                y_range = y1 - y0
                
                # Scale the Y movement to match X movement based on ranges
                # This ensures movement is proportional to the current view
                if x_range != 0:  # Prevent division by zero
                    dy = dy * (y_range / x_range)
                
                # Apply smoothing (exponential moving average)
                smoothing_factor = 0.7  # Adjust this value between 0 and 1 for different smoothing
                dx = dx * (1 - smoothing_factor) + self._last_dx * smoothing_factor
                dy = dy * (1 - smoothing_factor) + self._last_dy * smoothing_factor
                
                # Store current movement for next frame
                self._last_dx = dx
                self._last_dy = dy
                
                # Calculate new limits
                new_x0 = x0 - dx
                new_x1 = x1 - dx
                new_y0 = y0 - dy  # Fixed: removed inversion
                new_y1 = y1 - dy  # Fixed: removed inversion
                
                # Ensure minimum range to prevent singular transformation
                min_range = 1e-10  # Minimum range to prevent singular transformation
                if abs(new_x1 - new_x0) < min_range:
                    mid = (new_x0 + new_x1) / 2
                    new_x0 = mid - min_range/2
                    new_x1 = mid + min_range/2
                if abs(new_y1 - new_y0) < min_range:
                    mid = (new_y0 + new_y1) / 2
                    new_y0 = mid - min_range/2
                    new_y1 = mid + min_range/2
                
                # Update the view limits
                self.ax.set_xlim(new_x0, new_x1)
                self.ax.set_ylim(new_y0, new_y1)
                
                # Store the current mouse position in data coordinates
                self._last_mouse_pos_px = (current_x, current_y)
                
                # Only redraw the canvas, don't update the plot
                self.canvas.draw_idle()

    def on_scroll(self, event):
        if hasattr(self.ax, 'get_zlim'):
            # 3D plot: use previous logic
            if event.inaxes != self.ax:
                return
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            zlim = self.ax.get_zlim()
            zoom_factor = 1.1 if event.button == 'up' else 0.9
            self.ax.set_xlim(xlim[0] * zoom_factor, xlim[1] * zoom_factor)
            self.ax.set_ylim(ylim[0] * zoom_factor, ylim[1] * zoom_factor)
            self.ax.set_zlim(zlim[0] * zoom_factor, zlim[1] * zoom_factor)
            self.canvas.draw_idle()
        else:
            # 2D plot: improved zooming logic
            if event.inaxes:
                # Get current view limits
                x0, x1 = self.ax.get_xlim()
                y0, y1 = self.ax.get_ylim()
                
                # Get mouse position in data coordinates
                x, y = event.xdata, event.ydata
                
                # Calculate current ranges
                x_range = x1 - x0
                y_range = y1 - y0
                
                # Calculate zoom factor based on scroll direction
                # Use a smaller zoom factor for more controlled zooming
                # Fixed: inverted the zoom direction
                zoom_factor = 0.95 if event.step > 0 else 1.05
                
                # Calculate new ranges
                new_x_range = x_range * zoom_factor
                new_y_range = y_range * zoom_factor
                
                # Calculate new limits centered on mouse position
                # Use the ratio of mouse position to current range to maintain center point
                x_ratio = (x - x0) / x_range if x_range != 0 else 0.5
                y_ratio = (y - y0) / y_range if y_range != 0 else 0.5
                
                new_x0 = x - new_x_range * x_ratio
                new_x1 = x + new_x_range * (1 - x_ratio)
                new_y0 = y - new_y_range * y_ratio
                new_y1 = y + new_y_range * (1 - y_ratio)
                
                # Ensure minimum range to prevent singular transformation
                min_range = 1e-10
                if abs(new_x1 - new_x0) < min_range:
                    mid = (new_x0 + new_x1) / 2
                    new_x0 = mid - min_range/2
                    new_x1 = mid + min_range/2
                if abs(new_y1 - new_y0) < min_range:
                    mid = (new_y0 + new_y1) / 2
                    new_y0 = mid - min_range/2
                    new_y1 = mid + min_range/2
                
                # Update the view limits
                self.ax.set_xlim(new_x0, new_x1)
                self.ax.set_ylim(new_y0, new_y1)
                
                # Only redraw the canvas, don't update the plot
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
        # Update dimensionality if it's a valid input
        if self.optimization_type == "ND":
            try:
                dim_input = self.input_widget_groups["Спільні вхідні дані"].get_input("dimensionality")
                if dim_input.is_valid():
                    self.dimensionality = int(dim_input.get_data())
            except (RuntimeError, KeyError):
                pass
        self.update_plot()

    def show_load_data_window(self):
        load_window = SavedDataWindow(self)
        load_window.loaded_data = None
        load_window.exec()
        if hasattr(load_window, 'loaded_data') and load_window.loaded_data:
            self.apply_loaded_input_data(load_window.loaded_data)

    def apply_loaded_input_data(self, selected_data):
        # 1. Set optimization type
        if 'dimensionality' in selected_data and int(selected_data['dimensionality']) > 1:
            self.set_optimization_type("ND")
        else:
            self.set_optimization_type("1D")

        # 2. Force update of methods widget to create checkboxes
        self.update_methods_widget()
        QtWidgets.QApplication.processEvents()

        # 3. Check all method checkboxes for methods in file (first time)
        method_data = selected_data.get('method_data', {})
        for method_name in method_data:
            if method_name in self._methods_checkboxes:
                self._methods_checkboxes[method_name].setChecked(True)
        QtWidgets.QApplication.processEvents()

        # 4. Now update input parameters to create all input fields for checked methods
        self.update_input_parameters()
        QtWidgets.QApplication.processEvents()

        # 5. Check all method checkboxes for methods in file (second time, after possible widget rebuild)
        for method_name in method_data:
            if method_name in self._methods_checkboxes:
                self._methods_checkboxes[method_name].setChecked(True)
        QtWidgets.QApplication.processEvents()

        # 6. Fill all fields in 'Спільні вхідні дані' group
        common_group = self.input_widget_groups.get("Спільні вхідні дані")
        if common_group:
            for key, value in selected_data.items():
                if key in common_group.inputs:
                    input_widget = common_group.get_input(key)
                    if hasattr(input_widget, "line_edit"):
                        input_widget.line_edit.setText(str(value))
                    elif hasattr(input_widget, "set_values"):
                        input_widget.set_values(value)
                    elif hasattr(input_widget, "set_intervals"):
                        input_widget.set_intervals(value)
                    elif hasattr(input_widget, "set_checked"):
                        input_widget.set_checked(bool(value))

        # 7. Fill all method-specific fields
        for method_name, params in method_data.items():
            method_group = self.input_widget_groups.get(method_name)
            if method_group:
                for param_name, param_value in params.items():
                    input_widget = method_group.get_input(param_name)
                    if input_widget:
                        if hasattr(input_widget, "line_edit"):
                            input_widget.line_edit.setText(str(param_value))
                        elif hasattr(input_widget, "set_values"):
                            input_widget.set_values(param_value)
                        elif hasattr(input_widget, "set_intervals"):
                            input_widget.set_intervals(param_value)
                        elif hasattr(input_widget, "set_checked"):
                            input_widget.set_checked(bool(param_value))

    def apply_styles(self):
        self.setStyleSheet("""
            QWidget {
                font-family: Segoe UI, sans-serif;
                font-size: 14px;
                background-color: #f9f9fb;
                color: #222;
            }

            QMainWindow {
                background-color: #f0f2f5;
            }

            QGroupBox {
                border: 1px solid #d0d0d0;
                border-radius: 8px;
                margin-top: 16px;
                background-color: #ffffff;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #2c3e50;
                font-weight: bold;
                background-color: transparent;
            }

            QLabel {
                color: #34495e;
            }

            QPushButton {
                background-color: #2980b9;
                color: white;
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: 500;
            }

            QPushButton:hover {
                background-color: #3498db;
            }

            QPushButton:pressed {
                background-color: #2471a3;
            }

            QPushButton:checked {
                background-color: #1abc9c;
                color: white;
            }

            QLineEdit, QTextEdit {
                background-color: white;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                padding: 4px;
            }

            QLineEdit:focus, QTextEdit:focus {
                border: 1px solid #3498db;
                outline: none;
            }

            QCheckBox {
                padding: 4px;
            }

            QCheckBox {
                padding: 8px 4px;
                min-height: 20px;
                font-size: 15px;
                font-weight: 500;
            }

            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                margin-left: 4px;
                margin-right: 10px;
            }

            QCheckBox::indicator:checked {
                background-color: #3498db;
                image: url(none);
                border-radius: 3px;
            }

            QScrollArea {
                border: none;
                background: transparent;
            }

            QScrollBar:vertical {
                background: transparent;
                width: 12px;
                margin: 4px 0;
            }

            QScrollBar::handle:vertical {
                background: #bdc3c7;
                border-radius: 6px;
                min-height: 20px;
            }

            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }

            QFrame {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 6px;
            }

            QStackedLayout {
                background: transparent;
            }
        """)