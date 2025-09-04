import sympy
from PyQt6 import QtWidgets

class FunctionInput(QtWidgets.QWidget):
    def __init__(self, label_text: str, optimization_type, dimensionality, default_value: str = None, parent=None):
        super().__init__(parent)
        self.layout = QtWidgets.QHBoxLayout(self)
        self.label = QtWidgets.QLabel(label_text)
        self.line_edit = QtWidgets.QLineEdit(default_value)
        self.layout.addWidget(self.label, 1)
        self.layout.addWidget(self.line_edit, 1)
        self.setLayout(self.layout)

        self.optimization_type = optimization_type
        self.dimensionality = dimensionality

    def get_data(self):
        return self.line_edit.text()

    def is_valid(self):
        if not self.line_edit.text().strip():
            return False

        try:
            # Визначаємо дозволені змінні залежно від типу оптимізації
            if self.optimization_type == "1D":
                allowed_symbols = {sympy.Symbol("x")}
                if 'x' not in self.line_edit.text():
                    return False
            else:
                allowed_symbols = {sympy.Symbol(f"x{i + 1}") for i in range(self.dimensionality)}
                # Check if at least one of the allowed variables is present
                if not any(f"x{i+1}" in self.line_edit.text() for i in range(self.dimensionality)):
                    return False

            # Парсимо вираз без обчислення
            expr = sympy.sympify(self.line_edit.text(), evaluate=False)

            # Отримуємо фактичні змінні з виразу
            used_symbols = expr.free_symbols

            # Перевіряємо, що всі використані змінні — дозволені
            if not used_symbols.issubset(allowed_symbols):
                return False

            # Пробуємо підставити значення
            test_subs = {symbol: 1.0 for symbol in used_symbols}
            _ = expr.subs(test_subs)
            return True

        except Exception:
            return False

    def set_invalid(self):
        self.line_edit.setStyleSheet("border: 2px solid red;")

    def set_valid(self):
        self.line_edit.setStyleSheet("")

    def connect_text_changed(self, slot):
        self.line_edit.textChanged.connect(slot)

class FloatInput(QtWidgets.QWidget):
    def __init__(self, label_text: str, optimization_type, dimensionality, default_value: str = None, parent=None):
        super().__init__(parent)
        self.layout = QtWidgets.QHBoxLayout(self)
        self.label = QtWidgets.QLabel(label_text)
        self.line_edit = QtWidgets.QLineEdit(default_value)
        self.layout.addWidget(self.label, 1)
        self.layout.addWidget(self.line_edit, 1)
        self.setLayout(self.layout)

    def get_data(self):
        return float(self.line_edit.text())
    
    def is_valid(self):
        try: 
            float(self.line_edit.text())
            return True
        except Exception:
            return False

    def set_invalid(self):
        self.line_edit.setStyleSheet("border: 2px solid red;")

    def set_valid(self):
        self.line_edit.setStyleSheet("")

    def connect_text_changed(self, slot):
        self.line_edit.textChanged.connect(slot)

class DimensionalityInput(QtWidgets.QWidget):
    def __init__(self, label_text: str, optimization_type, dimensionality, default_value: str = None, parent=None):
        super().__init__(parent)
        self.layout = QtWidgets.QHBoxLayout(self)
        self.label = QtWidgets.QLabel(label_text)
        self.line_edit = QtWidgets.QLineEdit(default_value)
        self.layout.addWidget(self.label, 1)
        self.layout.addWidget(self.line_edit, 1)
        self.setLayout(self.layout)

    def get_data(self):
        return float(self.line_edit.text())

    def is_valid(self):
        try:
            value = int(self.line_edit.text())
            if value > 0:
                return True
            else:
                return False
        except ValueError:
            return False

    def set_invalid(self):
        self.line_edit.setStyleSheet("border: 2px solid red;")

    def set_valid(self):
        self.line_edit.setStyleSheet("")

    def connect_text_changed(self, slot):
        self.line_edit.textChanged.connect(slot)

class CheckboxInput(QtWidgets.QWidget):
    def __init__(self, label_text: str, optimization_type, dimensionality, default_value: str = None, parent=None):
        super().__init__(parent)
        self.layout = QtWidgets.QHBoxLayout(self)
        self.label = QtWidgets.QLabel(label_text)
        self.checkbox = QtWidgets.QCheckBox()
        self.checkbox.setChecked(True)
        self.layout.addWidget(self.label, 1)
        self.layout.addWidget(self.checkbox, 1)
        self.setLayout(self.layout)

    def is_valid(self):
        return True

    def get_data(self):
        return self.checkbox.isChecked()

    def set_checked(self, value: bool):
        self.checkbox.setChecked(value)

    def connect_state_changed(self, slot):
        self.checkbox.stateChanged.connect(slot)

class IntervalInput(QtWidgets.QWidget):
    def __init__(self, label_text: str, optimization_type, dimensionality, default_value: str = None, parent=None):
        super().__init__(parent)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.setLayout(self.layout)

        self.intervals = {}
        self._build_inputs(dimensionality)

    def _build_inputs(self, dimensionality: int):
        # Очистка попередніх елементів
        for i in reversed(range(self.layout.count())):
            child = self.layout.takeAt(i)
            if child.widget():
                child.widget().deleteLater()

        self.intervals.clear()

        # Створення рядків x1: від ___ до ___
        for i in range(dimensionality):
            var_name = f"x{i+1}"
            label = QtWidgets.QLabel(f"{var_name}: ")
            from_label = QtWidgets.QLabel("від")
            from_input = QtWidgets.QLineEdit()
            to_label = QtWidgets.QLabel("до")
            to_input = QtWidgets.QLineEdit()

            # Основний горизонтальний лейаут
            h_layout = QtWidgets.QHBoxLayout()
            h_layout.addWidget(label, 2)

            # Підлейаут "від ___"
            layout1 = QtWidgets.QHBoxLayout()
            layout1.addWidget(from_label)
            layout1.addWidget(from_input)
            layout1_container = QtWidgets.QWidget()
            layout1_container.setLayout(layout1)
            h_layout.addWidget(layout1_container, 1)

            # Підлейаут "до ___"
            layout2 = QtWidgets.QHBoxLayout()
            layout2.addWidget(to_label)
            layout2.addWidget(to_input)
            layout2_container = QtWidgets.QWidget()
            layout2_container.setLayout(layout2)
            h_layout.addWidget(layout2_container, 1)

            # Обгортка всієї лінії
            container = QtWidgets.QWidget()
            container.setLayout(h_layout)
            self.layout.addWidget(container)

            self.intervals[var_name] = (from_input, to_input)

    def is_valid(self):
        for var, (from_input, to_input) in self.intervals.items():
            try:
                a = float(from_input.text())
                b = float(to_input.text())
                if a > b:
                    raise ValueError
            except Exception:
                return False
        return True

    def get_data(self):
        """
        Повертає словник з інтервалами: {"x1": (від, до), "x2": (від, до), ...}
        Значення повертаються як рядки.
        """
        return {
            var: (float(inputs[0].text()), float(inputs[1].text()))
            for var, inputs in self.intervals.items()
        }

    def set_intervals(self, values: dict):
        """
        Приймає словник {"x1": ("1", "10"), ...} і встановлює значення в поля.
        """
        for var, (from_val, to_val) in values.items():
            if var in self.intervals:
                self.intervals[var][0].setText(str(from_val))
                self.intervals[var][1].setText(str(to_val))

    def set_invalid(self, var: str = 0):
        """
        Підсвічує інтервал для певного виміру червоним.
        """
        if var in self.intervals:
            self.intervals[var][0].setStyleSheet("border: 2px solid red;")
            self.intervals[var][1].setStyleSheet("border: 2px solid red;")

    def set_valid(self, var: str = 0):
        """
        Скидає підсвічування для певного виміру.
        """
        if var in self.intervals:
            self.intervals[var][0].setStyleSheet("")
            self.intervals[var][1].setStyleSheet("")

class VectorInput(QtWidgets.QWidget):
    def __init__(self, label_text: str, optimization_type, dimensionality, default_value: str = None, parent=None):
        super().__init__(parent)
        self.layout = QtWidgets.QHBoxLayout(self)
        self.label = QtWidgets.QLabel(label_text)
        self.inputs_layout = QtWidgets.QHBoxLayout()

        self.inputs = []

        if default_value is None:
            default_value = [""] * dimensionality
        elif len(default_value) < dimensionality:
            default_value += [""] * (dimensionality - len(default_value))

        for i in range(dimensionality):
            line_edit = QtWidgets.QLineEdit(default_value[i])
            line_edit.setFixedWidth(50)
            self.inputs.append(line_edit)
            self.inputs_layout.addWidget(line_edit)
            if i < dimensionality - 1:
                self.inputs_layout.addWidget(QtWidgets.QLabel(";"))

        self.layout.addWidget(self.label, 1)
        self.layout.addLayout(self.inputs_layout, 1)
        self.setLayout(self.layout)

    def is_valid(self):
        for line in self.inputs:
            try:
                float(line.text())
            except ValueError:
                return False
        return True

    def get_data(self):
        return [float(line.text()) for line in self.inputs]

    def set_values(self, values):
        for line, value in zip(self.inputs, values):
            line.setText(str(value))

    def connect_text_changed(self, slot):
        for line in self.inputs:
            line.textChanged.connect(slot)

    def set_invalid(self):
        for line in self.inputs:
            try:
                float(line.text())
                line.setStyleSheet("")
            except ValueError:
                line.setStyleSheet("border: 2px solid red;")

    def set_valid(self):
        self.set_invalid()

class LabeledGroup(QtWidgets.QGroupBox):
    def __init__(self, title: str = "Група даних", parent=None):
        super().__init__(title, parent)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.setLayout(self.layout)
        self.inputs = {}
        
    def validate_all(self):
        valid = True
        for widget in self.inputs.values():
            if hasattr(widget, "is_valid") and not widget.is_valid():
                valid = False
        return valid

    def add_input(self, key: str, input_widget: QtWidgets.QWidget):
        """Додає іменований віджет до групи"""
        self.layout.addWidget(input_widget)
        self.inputs[key] = input_widget

    def get_input(self, key: str):
        """Повертає віджет за ключем"""
        return self.inputs.get(key)

    def clear_inputs(self):
        """Очищає всі віджети в групі"""
        while self.layout.count():
            child = self.layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.inputs.clear()

def input_identifier_to_widget(identifier):
    return {
        "function": {
            "is_necessary": True,
            "label": "Цільова функція",
            "type_of_input": FunctionInput
        },
        "dimensionality" : {
            "is_necessary": True,
            "label": "Кількість вимірів",
            "type_of_input": DimensionalityInput
        },
        "error": {
            "is_necessary": True,
            "label": "Похибка (ε)",
            "type_of_input": FloatInput
        },
        "max_iterations": {
            "is_necessary": False,
            "label": "Максимальна кількість ітерацій",
            "type_of_input": FloatInput
        },
        "n_steps": {
            "is_necessary": False,
            "label": "Кількість кроків",
            "type_of_input": FloatInput
        },
        "alpha": {
            "is_necessary": False,
            "label": "Коефіцієнт α",
            "type_of_input": FloatInput
        },
        "gamma": {
            "is_necessary": False,
            "label": "Коефіцієнт γ",
            "type_of_input": FloatInput
        },
        "rho": {
            "is_necessary": False,
            "label": "Коефіцієнт ρ",
            "type_of_input": FloatInput
        },
        "sigma": {
            "is_necessary": False,
            "label": "Коефіцієнт σ",
            "type_of_input": FloatInput
        },
        "initial_step": {
            "is_necessary": False,
            "label": "Початковий крок",
            "type_of_input": FloatInput
        },
        "step_reduction": {
            "is_necessary": False,
            "label": "Коефіцієнт зменшення кроку",
            "type_of_input": FloatInput
        },
        "lambda": {
            "is_necessary": False,
            "label": "Коефіцієнт λ",
            "type_of_input": FloatInput
        },
        "population_size": {
            "is_necessary": False,
            "label": "Розмір популяції",
            "type_of_input": FloatInput
        },
        "F": {
            "is_necessary": False,
            "label": "Коефіцієнт F (масштаб мутації)",
            "type_of_input": FloatInput
        },
        "CR": {
            "is_necessary": False,
            "label": "Ймовірність кросоверу (CR)",
            "type_of_input": FloatInput
        },
        "initial_temperature": {
            "is_necessary": False,
            "label": "Початкова температура",
            "type_of_input": FloatInput
        },
        "cooling_rate": {
            "is_necessary": False,
            "label": "Швидкість охолодження",
            "type_of_input": FloatInput
        },
        "is_max": {
            "is_necessary": False,
            "label": "Максимізація (обрати, якщо так)",
            "type_of_input": CheckboxInput
        },
        "interval": {
            "is_necessary": False,
            "label": "Інтервали змінних",
            "type_of_input": IntervalInput
        },
        "bounds": {
            "is_necessary": False,
            "label": "Межі області пошуку",
            "type_of_input": IntervalInput
        },
        "start_point": {
            "is_necessary": False,
            "label": "Початкова точка",
            "type_of_input": VectorInput
        }
    }[identifier]

def build_input_groups(grouped_identifiers: dict, optimization_type: str, dimensionality: int, parent) -> dict:
    result = {}

    for group_name, identifiers in grouped_identifiers.items():
        group = LabeledGroup(group_name, parent)

        for identifier in identifiers:
            try:
                widget_info = input_identifier_to_widget(identifier)
                widget_class = widget_info["type_of_input"]

                widget = widget_class(widget_info["label"], optimization_type, dimensionality, None, group)

                group.add_input(identifier, widget)
            except KeyError:
                print(f"Ідентифікатор '{identifier}' не знайдено у конфігурації input_identifier_to_widget.")
            except Exception as e:
                print(f"Помилка при створенні поля '{identifier}': {e}")

        result[group_name] = group

    return result