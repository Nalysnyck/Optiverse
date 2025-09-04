import time, numpy, sympy

from logic.categories import *
from controllers.function import *

class OptimizationMethodBase:
    def __init__(self):
        self.name = None
        self.category = None
        self.input_identifiers = []

        self.result = {
            "time": 0,
            "iterations": {},
            "optimum": {
                "position": {},
                "value": None
            },
            "is_successful": None,
            "failure_reason": None
        }

    def run(self, input: dict):
        start_time = time.time()

        self._execute(input)

        self.result["time"] = time.time() - start_time

    def _execute(self, input: dict):
        raise NotImplementedError("Subclasses must implement 'execute()'")

class GoldenRatioMethod(OptimizationMethodBase):
    def __init__(self):
        super().__init__()

        self.name = 'Метод золотого перерізу'
        self.category = OneDimensionalOptimizationCategory()
        self.iteration_result_key = "c"
    
    def _execute(self, input: dict):
        iteration = 0
        a, b = input["interval"]
        c, d = a, a
        gr_koef_1 = -(1 - 5 ** 0.5) / 2
        gr_koef_2 = (3 - 5 ** 0.5) / 2

        while True:
            c = a + (b - a) * gr_koef_2
            d = a + (b - a) * gr_koef_1

            cur_error = abs(d - c)

            self.result["iterations"][iteration] = {
                "a": a, 
                "b": b, 
                "c": c, 
                "d": d, 
                "f(c)": input["function"]([c]), 
                "f(d)": input["function"]([d]), 
                "error": cur_error
            }
            iteration += 1

            if cur_error < input["error"]:
                self.result["is_successful"] = True
                optimum = (c + d) / 2
                self.result["optimum"] = {
                    "position": {"x": optimum},
                    "value": input["function"]([optimum])
                }
                break

            if iteration >= input["max_iterations"]:
                self.result["failure_reason"] = "Maximum iterations reached."
                self.result["is_successful"] = False
                break
            
            difference = input["function"]([d]) - input["function"]([c])

            if difference > 0 and input["is_max"] or difference < 0 and not input["is_max"]:
                a = c
            else:
                b = d

    def get_iteration_descriptions(self):
        descriptions = []
        iterations = self.result.get("iterations", {})
        iter_keys = sorted(iterations.keys(), key=lambda x: int(x) if isinstance(x, str) and x.isdigit() else x)
        for i, key in enumerate(iter_keys):
            it = iterations[key]
            a = it["a"]
            b = it["b"]
            c = it["c"]
            d = it["d"]
            fc = it["f(c)"]
            fd = it["f(d)"]
            error = it["error"]
            if i + 1 < len(iter_keys):
                next_it = iterations[iter_keys[i+1]]
                next_a = next_it["a"]
                next_b = next_it["b"]
            else:
                next_a = a
                next_b = b
            desc = (
                f"Інтервал: [{a:.3f}; {b:.3f}]\n"
                f"Обчислюємо точки c і d:\n"
                f"c = {c:.3f}, d = {d:.3f}\n"
                f"f(c) = {fc:.5f}, f(d) = {fd:.5f}\n"
                f"Наступний інтервал: [{next_a:.3f}; {next_b:.3f}]\n"
                f"Похибка: {error:.5f}"
            )
            descriptions.append(desc)
        return descriptions

class DichotomyMethod(OptimizationMethodBase):
    def __init__(self):
        super().__init__()

        self.name = 'Метод половинного поділу'
        self.category = OneDimensionalOptimizationCategory()
        self.iteration_result_key = "c"

    def _execute(self, input: dict):
        iteration = 0
        a, b = input["interval"]
        c = 0

        while True:
            c = (a + b) / 2

            cur_error = abs(a - b)

            self.result["iterations"][iteration] = {
                "a": a, 
                "b": b, 
                "c": c, 
                "f(a)": input["function"]([a]), 
                "f(b)": input["function"]([b]), 
                "f(c)": input["function"]([c]), 
                "error": cur_error
            }
            iteration += 1

            if cur_error < input["error"]:
                self.result["is_successful"] = True
                self.result["optimum"] = {
                    "position": {"x": c},
                    "value": input["function"]([c])
                }
                break

            if iteration >= input["max_iterations"]:
                self.result["failure_reason"] = "Maximum iterations reached."
                self.result["is_successful"] = False
                break

            difference = input["function"]([b]) - input["function"]([a])
            
            if difference > 0 and input["is_max"] or difference < 0 and not input["is_max"]:
                a = c
            else:
                b = c

    def get_iteration_descriptions(self):
        descriptions = []
        iterations = self.result.get("iterations", {})
        iter_keys = sorted(iterations.keys(), key=lambda x: int(x) if isinstance(x, str) and x.isdigit() else x)
        for i, key in enumerate(iter_keys):
            it = iterations[key]
            a = it["a"]
            b = it["b"]
            c = it["c"]
            fa = it["f(a)"]
            fb = it["f(b)"]
            fc = it["f(c)"]
            error = it["error"]
            if i + 1 < len(iter_keys):
                next_it = iterations[iter_keys[i+1]]
                next_a = next_it["a"]
                next_b = next_it["b"]
            else:
                next_a = a
                next_b = b
            desc = (
                f"Інтервал: [{a:.1f}; {b:.1f}]\n"
                f"Ділимо інтервал навпіл:\n"
                f"c = ({a:.1f} + {b:.1f}) / 2 = {c:.1f}\n"
                f"f(a) = {fa:.2f}\n"
                f"f(b) = {fb:.2f}\n"
                f"f(c) = {fc:.2f}\n"
                f"Наступний інтервал: [{next_a:.1f}; {next_b:.1f}]\n"
                f"Похибка: {error:.0f}"
            )
            descriptions.append(desc)
        return descriptions

class TernarySearchMethod(OptimizationMethodBase):
    def __init__(self):
        super().__init__()

        self.name = 'Метод тернарного пошуку'
        self.category = OneDimensionalOptimizationCategory()
        self.iteration_result_key = "left_third"

    def _execute(self, input: dict):
        iteration = 0
        a, b = input["interval"]

        while True:
            cur_error = abs(a - b)

            # Calculate two middle points
            left_third = a + (b - a) / 3
            right_third = b - (b - a) / 3

            f_left = input["function"]([left_third])
            f_right = input["function"]([right_third])

            # Store iteration data
            self.result["iterations"][iteration] = {
                "a": a, 
                "b": b, 
                "left_third": left_third, 
                "right_third": right_third,
                "f(left_third)": f_left, 
                "f(right_third)": f_right, 
                "error": cur_error
            }
            iteration += 1

            # Check for termination condition
            if cur_error < input["error"]:
                self.result["is_successful"] = True
                optimum = (left_third + right_third) / 2
                self.result["optimum"] = {
                    "position": {"x": optimum},
                    "value": input["function"]([optimum])
                }
                break

            if iteration >= input["max_iterations"]:
                self.result["is_successful"] = False
                break

            # Update interval based on objective (max/min)
            if (f_left < f_right and not input["is_max"]) or (f_left > f_right and input["is_max"]):
                b = right_third
            else:
                a = left_third

    def get_iteration_descriptions(self):
        descriptions = []
        iterations = self.result.get("iterations", {})
        iter_keys = sorted(iterations.keys(), key=lambda x: int(x) if isinstance(x, str) and x.isdigit() else x)
        for i, key in enumerate(iter_keys):
            it = iterations[key]
            a = it["a"]
            b = it["b"]
            left = it["left_third"]
            right = it["right_third"]
            f_left = it["f(left_third)"]
            f_right = it["f(right_third)"]
            error = it["error"]
            if i + 1 < len(iter_keys):
                next_it = iterations[iter_keys[i+1]]
                next_a = next_it["a"]
                next_b = next_it["b"]
            else:
                next_a = a
                next_b = b
            desc = (
                f"Інтервал: [{a:.3f}; {b:.3f}]\n"
                f"Ділимо інтервал на третини:\n"
                f"left = {left:.3f}, right = {right:.3f}\n"
                f"f(left) = {f_left:.5f}\n"
                f"f(right) = {f_right:.5f}\n"
                f"Наступний інтервал: [{next_a:.3f}; {next_b:.3f}]\n"
                f"Похибка: {error:.5f}"
            )
            descriptions.append(desc)
        return descriptions

class FibonacciMethod(OptimizationMethodBase):
    def __init__(self):
        super().__init__()

        self.name = 'Метод Фібоначі'
        self.category = OneDimensionalOptimizationCategory()
        self.iteration_result_key = "c"

        self.__fibonacci = [1, 1]

    def _execute(self, input: dict):
        iteration = 0
        a, b = input["interval"]
        n = 0

        n_of_fibonacci_steps = (b - a) / input["error"]

        while self.__fibonacci[n] <= n_of_fibonacci_steps:
            n += 1
            if n > len(self.__fibonacci) - 1:
                self.__fibonacci.append(self.__fibonacci[-1] + self.__fibonacci[-2])
        
        cur_n = n

        for i in range(n - 1):
            cur_step = (b - a) / self.__fibonacci[cur_n]

            c = a + self.__fibonacci[cur_n - 2] * cur_step
            d = a + self.__fibonacci[cur_n - 1] * cur_step

            cur_error = abs(a - b)

            self.result["iterations"][iteration] = {
                "a": a, 
                "b": b, 
                "c": c, 
                "d": d, 
                "f(a)": input["function"]([a]), 
                "f(b)": input["function"]([b]), 
                "f(c)": input["function"]([c]), 
                "f(d)": input["function"]([d]), 
                "error": cur_error
            }
            iteration += 1

            if input["function"]([c]) < input["function"]([d]) and input["is_max"] or input["function"]([c]) > input["function"]([d]) and not input["is_max"]:
                a = c
            else:
                b = d
            
            cur_n -= 1
        
        self.result["is_successful"] = True
        optimum = (c + d) / 2
        self.result["optimum"] = {
            "position": {"x": optimum},
            "value": input["function"]([optimum])
        }
    def get_iteration_descriptions(self):
        descriptions = []
        iterations = self.result.get("iterations", {})
        iter_keys = sorted(iterations.keys(), key=lambda x: int(x) if isinstance(x, str) and x.isdigit() else x)
        for i, key in enumerate(iter_keys):
            it = iterations[key]
            a = it["a"]
            b = it["b"]
            c = it["c"]
            d = it["d"]
            fa = it["f(a)"]
            fb = it["f(b)"]
            fc = it["f(c)"]
            fd = it["f(d)"]
            error = it["error"]
            if i + 1 < len(iter_keys):
                next_it = iterations[iter_keys[i+1]]
                next_a = next_it["a"]
                next_b = next_it["b"]
            else:
                next_a = a
                next_b = b
            desc = (
                f"Інтервал: [{a:.3f}; {b:.3f}]\n"
                f"Обчислюємо точки c і d:\n"
                f"c = {c:.3f}, d = {d:.3f}\n"
                f"f(a) = {fa:.5f}, f(b) = {fb:.5f}\n"
                f"f(c) = {fc:.5f}, f(d) = {fd:.5f}\n"
                f"Наступний інтервал: [{next_a:.3f}; {next_b:.3f}]\n"
                f"Похибка: {error:.5f}"
            )
            descriptions.append(desc)
        return descriptions

class NewtonRaphsonMethod(OptimizationMethodBase):
    def __init__(self):
        super().__init__()
        self.name = 'Метод Ньютона–Рафсона'
        self.category = OneDimensionalOptimizationCategory()
        self.iteration_result_key = "x"
        self.input_identifiers = ["start_point"]

    def _execute(self, input: dict):
        iteration = 0
        x = input["start_point"][0]
        error_threshold = input["error"]
        max_iterations = input["max_iterations"]

        while iteration < max_iterations:
            grad = input["derivative1"]([x])
            hessian = input["derivative2"]([x])

            if hessian == 0:
                self.result["failure_reason"] = "Zero second derivative (division by zero)."
                self.result["is_successful"] = False
                break

            previous_x = x
            x = x - grad / hessian  # Стандартна формула Ньютона для екстремумів
            cur_error = abs(x - previous_x)

            self.result["iterations"][iteration] = {
                "x_prev": previous_x,
                "x": x,
                "f'(x)": grad,
                "f''(x)": hessian,
                "error": cur_error
            }

            if cur_error <= error_threshold:
                # Перевірка: це максимум чи мінімум?
                is_max = input["is_max"]
                if (is_max and hessian < 0) or (not is_max and hessian > 0):
                    self.result["is_successful"] = True
                    self.result["optimum"] = {
                        "position": {"x": x},
                        "value": input["function"]([x])
                    }
                    break
                else:
                    self.result["is_successful"] = False
                    self.result["failure_reason"] = (
                        "Second derivative does not indicate expected extremum type."
                    )
                    break

            iteration += 1

        if iteration >= max_iterations:
            self.result["failure_reason"] = "Maximum iterations reached."
            self.result["is_successful"] = False

    def get_iteration_descriptions(self):
        descriptions = []
        iterations = self.result.get("iterations", {})
        iter_keys = sorted(iterations.keys(), key=lambda x: int(x) if isinstance(x, str) and x.isdigit() else x)
        for i, key in enumerate(iter_keys):
            it = iterations[key]
            x_prev = it["x_prev"]
            x = it["x"]
            grad = it["f'(x)"]
            hessian = it["f''(x)"]
            error = it["error"]
            desc = (
                f"Ітерація {i+1}:\n"
                f"x_prev = {x_prev:.5f}\n"
                f"f'(x) = {grad:.5f}\n"
                f"f''(x) = {hessian:.5f}\n"
                f"x_new = x_prev - f'(x)/f''(x) = {x:.5f}\n"
                f"Похибка: {error:.5f}"
            )
            descriptions.append(desc)
        return descriptions

class ExhaustiveSearchMethod(OptimizationMethodBase):
    def __init__(self):
        super().__init__()

        self.name = 'Метод загального перебору'
        self.category = OneDimensionalOptimizationCategory()
        self.iteration_result_key = "x"
        self.input_identifiers = ["n_steps"]
    
    def _execute(self, input: dict):
        iteration = 0
        a, b = input["interval"]
        step = (b - a) / input["n_steps"]
        x = []
        f = []
        cur_error = abs(a - b)

        while cur_error >= input["error"]:
            for i in range(int(round((b - a) / step)) + 1):
                x.append(a + i * step)
                f.append(input["function"]([x[i]]))

            optimal_x = x[f.index(max(f) if input["is_max"] else min(f))]

            new_a, new_b = optimal_x, optimal_x

            if optimal_x - step <= a:
                new_b += step
            elif optimal_x + step >= b:
                new_a -= step
            else:
                if input["function"]([optimal_x - step]) <= input["function"]([optimal_x + step]) and input["is_max"] or input["function"]([optimal_x - step]) > input["function"]([optimal_x + step]) and not input["is_max"]:
                    new_a -= step
                else:
                    new_b += step

            cur_error = abs(a - b)

            self.result["iterations"][iteration] = {"a": a, "b": b, "x": optimal_x, "f(x)": input["function"]([optimal_x]), "error": cur_error}
            iteration += 1

            a, b = new_a, new_b

            x.clear()
            f.clear()
            step /= input["n_steps"]
        
        self.result["is_successful"] = True
        optimum = (a + b) / 2
        self.result["optimum"] = {
            "position": {"x": optimum},
            "value": input["function"]([optimum])
        }
        
    def get_iteration_descriptions(self):
        descriptions = []
        iterations = self.result.get("iterations", {})
        iter_keys = sorted(iterations.keys(), key=lambda x: int(x) if isinstance(x, str) and x.isdigit() else x)
        for i, key in enumerate(iter_keys):
            it = iterations[key]
            a = it["a"]
            b = it["b"]
            x = it["x"]
            fx = it["f(x)"]
            error = it["error"]
            if i + 1 < len(iter_keys):
                next_it = iterations[iter_keys[i+1]]
                next_a = next_it["a"]
                next_b = next_it["b"]
            else:
                next_a = a
                next_b = b
            desc = (
                f"Інтервал: [{a:.3f}; {b:.3f}]\n"
                f"Загальний перебір: x = {x:.3f}\n"
                f"f(x) = {fx:.5f}\n"
                f"Наступний інтервал: [{next_a:.3f}; {next_b:.3f}]\n"
                f"Похибка: {error:.5f}"
            )
            descriptions.append(desc)
        return descriptions

    def get_iteration_descriptions(self):
        descriptions = []
        iterations = self.result.get("iterations", {})
        iter_keys = sorted(iterations.keys(), key=lambda x: int(x) if isinstance(x, str) and x.isdigit() else x)
        for i, key in enumerate(iter_keys):
            it = iterations[key]
            a = it["a"]
            b = it["b"]
            x = it["x"]
            fx = it["f(x)"]
            error = it["error"]
            if i + 1 < len(iter_keys):
                next_it = iterations[iter_keys[i+1]]
                next_a = next_it["a"]
                next_b = next_it["b"]
            else:
                next_a = a
                next_b = b
            desc = (
                f"Інтервал: [{a:.3f}; {b:.3f}]\n"
                f"Загальний перебір: x = {x:.3f}\n"
                f"f(x) = {fx:.5f}\n"
                f"Наступний інтервал: [{next_a:.3f}; {next_b:.3f}]\n"
                f"Похибка: {error:.5f}"
            )
            descriptions.append(desc)
        return descriptions

class PowellsUnidimensionalMethod(OptimizationMethodBase):
    def __init__(self):
        super().__init__()

        self.name = 'Метод Пауелла'
        self.category = OneDimensionalOptimizationCategory()
        self.iteration_result_key = "x1"

    def _execute(self, input: dict):
        iteration = 0
        a, b = input["interval"]
        alpha = 0.618  # Commonly used value in Powell's method
        epsilon = input["error"]
        max_iterations = input["max_iterations"]

        while iteration < max_iterations:
            x1 = a + (1 - alpha) * (b - a)
            x2 = a + alpha * (b - a)

            f1 = input["function"]([x1])
            f2 = input["function"]([x2])

            cur_error = abs(b - a)

            # Store iteration data
            self.result["iterations"][iteration] = {
                "a": a,
                "b": b,
                "x1": x1,
                "x2": x2,
                "f(x1)": f1,
                "f(x2)": f2,
                "error": cur_error
            }
            iteration += 1

            if cur_error < epsilon:
                self.result["is_successful"] = True
                optimum = (x1 + x2) / 2
                self.result["optimum"] = {
                    "position": {"x": optimum},
                    "value": input["function"]([optimum])
                }
                break

            if (f1 < f2 and not input["is_max"]) or (f1 > f2 and input["is_max"]):
                b = x2
            else:
                a = x1

        # Check if max iterations reached without convergence
        if iteration >= max_iterations:
            self.result["is_successful"] = False
            self.result["failure_reason"] = "Maximum iterations reached without convergence"
            
    def get_iteration_descriptions(self):
        descriptions = []
        iterations = self.result.get("iterations", {})
        iter_keys = sorted(iterations.keys(), key=lambda x: int(x) if isinstance(x, str) and x.isdigit() else x)
        for i, key in enumerate(iter_keys):
            it = iterations[key]
            a = it["a"]
            b = it["b"]
            x1 = it["x1"]
            x2 = it["x2"]
            f1 = it["f(x1)"]
            f2 = it["f(x2)"]
            error = it["error"]
            if i + 1 < len(iter_keys):
                next_it = iterations[iter_keys[i+1]]
                next_a = next_it["a"]
                next_b = next_it["b"]
            else:
                next_a = a
                next_b = b
            desc = (
                f"Інтервал: [{a:.3f}; {b:.3f}]\n"
                f"Обчислюємо точки x1 і x2:\n"
                f"x1 = {x1:.3f}, x2 = {x2:.3f}\n"
                f"f(x1) = {f1:.5f}\n"
                f"f(x2) = {f2:.5f}\n"
                f"Наступний інтервал: [{next_a:.3f}; {next_b:.3f}]\n"
                f"Похибка: {error:.5f}"
            )
            descriptions.append(desc)
        return descriptions

class CubicSearchMethod(OptimizationMethodBase):
    def __init__(self):
        super().__init__()
        self.name = 'Метод кубічної апроксимації'
        self.category = OneDimensionalOptimizationCategory()
        self.iteration_result_key = "x_opt"

    def _execute(self, input: dict):
        iteration = 0
        a, b = input['interval']
        epsilon = input['error']
        is_max = input['is_max']

        while iteration < input['max_iterations']:
            f_a = input['function']([a])
            f_b = input['function']([b])

            # Похідні через скінченні різниці
            f_a_prime = (input['function']([a + epsilon]) - f_a) / epsilon
            f_b_prime = (input['function']([b + epsilon]) - f_b) / epsilon

            # Обчислення коефіцієнтів кубічної апроксимації
            denominator = b - a
            if denominator == 0:
                self.result['failure_reason'] = 'Zero interval width.'
                self.result['is_successful'] = False
                return

            z = 3 * (f_b - f_a) / denominator - (f_b_prime + 2 * f_a_prime)
            sqrt_term = (f_b_prime - f_a_prime)**2 - z**2

            if sqrt_term < 0:
                self.result['failure_reason'] = 'Negative root in cubic approximation.'
                self.result['is_successful'] = False
                return

            w = sqrt_term ** 0.5
            mu = (f_b_prime + w - z) / (2 * w) if w != 0 else 0.5
            mu = max(0, min(1, mu))

            x_opt = a + mu * (b - a)
            f_opt = input['function']([x_opt])
            cur_error = abs(b - a)

            self.result['iterations'][iteration] = {
                'a': a,
                'b': b,
                'x_opt': x_opt,
                'f(a)': f_a,
                'f(b)': f_b,
                'f(x_opt)': f_opt,
                'error': cur_error
            }
            iteration += 1

            if cur_error < epsilon:
                self.result['is_successful'] = True
                self.result['optimum'] = {
                    'position': {'x': x_opt},
                    'value': f_opt
                }
                return

            # Оновлення інтервалу за значеннями функції
            if (is_max and f_opt > f_b) or (not is_max and f_opt < f_b):
                a = x_opt
            else:
                b = x_opt

        # Якщо не завершено раніше — перевірка
        self.result['failure_reason'] = 'Maximum iterations reached.'
        self.result['is_successful'] = False

    def get_iteration_descriptions(self):
        descriptions = []
        iterations = self.result.get("iterations", {})
        iter_keys = sorted(iterations.keys(), key=lambda x: int(x) if isinstance(x, str) and x.isdigit() else x)
        for i, key in enumerate(iter_keys):
            it = iterations[key]
            a = it["a"]
            b = it["b"]
            x_opt = it["x_opt"]
            fa = it["f(a)"]
            fb = it["f(b)"]
            f_opt = it["f(x_opt)"]
            error = it["error"]
            if i + 1 < len(iter_keys):
                next_it = iterations[iter_keys[i+1]]
                next_a = next_it["a"]
                next_b = next_it["b"]
            else:
                next_a = a
                next_b = b
            desc = (
                f"Інтервал: [{a:.3f}; {b:.3f}]\n"
                f"Обчислюємо оптимальну точку:\n"
                f"x_opt = {x_opt:.3f}\n"
                f"f(a) = {fa:.5f}, f(b) = {fb:.5f}\n"
                f"f(x_opt) = {f_opt:.5f}\n"
                f"Наступний інтервал: [{next_a:.3f}; {next_b:.3f}]\n"
                f"Похибка: {error:.5f}"
            )
            descriptions.append(desc)
        return descriptions

class SecantMethod(OptimizationMethodBase):
    def __init__(self):
        super().__init__()
        self.name = 'Метод хорд'
        self.category = OneDimensionalOptimizationCategory()
        self.iteration_result_key = "x_new"

    def _execute(self, input: dict):
        iteration = 0
        x0, x1 = input["interval"]
        tol = input["error"]
        max_iter = input["max_iterations"]
        is_max = input["is_max"]

        while iteration < max_iter:
            f1_x0 = input["derivative1"]([x0])
            f1_x1 = input["derivative1"]([x1])

            denominator = f1_x1 - f1_x0
            if abs(denominator) < 1e-12:
                self.result["is_successful"] = False
                self.result["failure_reason"] = "Division by near-zero in secant calculation."
                break

            x_new = x1 - f1_x1 * (x1 - x0) / denominator
            f1_x_new = input["derivative1"]([x_new])
            f2_x_new = input["derivative2"]([x_new])
            f_x_new = input["function"]([x_new])
            cur_error = abs(x_new - x1)

            self.result["iterations"][iteration] = {
                "x0": x0,
                "x1": x1,
                "x_new": x_new,
                "f'(x0)": f1_x0,
                "f'(x1)": f1_x1,
                "f'(x_new)": f1_x_new,
                "f''(x_new)": f2_x_new,
                "f(x_new)": f_x_new,
                "error": cur_error
            }

            # Умова зупинки — досягли похибки + правильного напрямку (опуклості)
            if cur_error < tol and (
                (is_max and f2_x_new < 0) or (not is_max and f2_x_new > 0)
            ):
                self.result["is_successful"] = True
                self.result["optimum"] = {
                    "position": {"x": x_new},
                    "value": f_x_new
                }
                break

            # Якщо знайшли "не той" екстремум, рухаємось в іншу сторону
            if (is_max and f2_x_new > 0) or (not is_max and f2_x_new < 0):
                # Відкидаємо цей напрям і трохи зміщуємось
                x_new = (x0 + x1) / 2

            x0, x1 = x1, x_new
            iteration += 1

        if iteration >= max_iter:
            self.result["is_successful"] = False
            self.result["failure_reason"] = "Maximum iterations reached."

    def get_iteration_descriptions(self):
        descriptions = []
        iterations = self.result.get("iterations", {})
        iter_keys = sorted(iterations.keys(), key=lambda x: int(x) if isinstance(x, str) and x.isdigit() else x)
        for i, key in enumerate(iter_keys):
            it = iterations[key]
            x0 = it["x0"]
            x1 = it["x1"]
            x_new = it["x_new"]
            f1_x0 = it["f'(x0)"]
            f1_x1 = it["f'(x1)"]
            f1_x_new = it["f'(x_new)"]
            f2_x_new = it["f''(x_new)"]
            f_x_new = it["f(x_new)"]
            error = it["error"]
            desc = (
                f"Поточні точки: x0 = {x0:.3f}, x1 = {x1:.3f}\n"
                f"Нова точка: x_new = {x_new:.3f}\n"
                f"Похідні в точках:\n"
                f"f'(x0) = {f1_x0:.5f}\n"
                f"f'(x1) = {f1_x1:.5f}\n"
                f"f'(x_new) = {f1_x_new:.5f}\n"
                f"f''(x_new) = {f2_x_new:.5f}\n"
                f"f(x_new) = {f_x_new:.5f}\n"
                f"Похибка: {error:.5f}"
            )
            descriptions.append(desc)
        return descriptions

class BrentsMethod(OptimizationMethodBase):
    def __init__(self):
        super().__init__()
        self.name = 'Метод Брента'
        self.category = OneDimensionalOptimizationCategory()
        self.iteration_result_key = "s"

    def _execute(self, input: dict):
        a, b = input['interval']
        fa = input['function']([a])
        fb = input['function']([b])

        if fa * fb > 0:
            self.result['failure_reason'] = 'The function must have opposite signs at the interval boundaries.'
            self.result['is_successful'] = False
            return

        tol = input['error']
        max_iter = input['max_iterations']
        iteration = 0

        c, d, e = a, a, a
        fc, fd, fe = fa, fa, fa
        mflag = True

        while iteration < max_iter:
            iteration += 1

            if fa != fb and fb != fc and fa != fc:
                # Inverse quadratic interpolation
                L0 = (a * fb * fc) / ((fa - fb) * (fa - fc))
                L1 = (b * fa * fc) / ((fb - fa) * (fb - fc))
                L2 = (c * fa * fb) / ((fc - fa) * (fc - fb))
                s = L0 + L1 + L2
            else:
                # Secant method
                s = b - fb * (b - a) / (fb - fa)

            cond1 = (s < (3 * a + b) / 4 or s > b) if a < b else (s > (3 * a + b) / 4 or s < b)
            cond2 = mflag and abs(s - b) >= abs(b - c) / 2
            cond3 = not mflag and abs(s - b) >= abs(c - d) / 2
            cond4 = mflag and abs(b - c) < tol
            cond5 = not mflag and abs(c - d) < tol

            if cond1 or cond2 or cond3 or cond4 or cond5:
                # Bisection method
                s = (a + b) / 2
                mflag = True
            else:
                mflag = False

            fs = input['function']([s])
            d, e = c, b
            fd, fe = fc, fb
            c, fc = b, fb

            if fa * fs < 0:
                b = s
                fb = fs
            else:
                a = s
                fa = fs

            if abs(b - a) < tol:
                self.result['is_successful'] = True
                self.result['optimum'] = {'position': {'x': s}, 'value': fs}
                break

            self.result['iterations'][iteration] = {
                'a': a,
                'b': b,
                's': s,
                'f(a)': fa,
                'f(b)': fb,
                'f(s)': fs,
                'error': abs(b - a)
            }

        if iteration >= max_iter:
            self.result['failure_reason'] = 'Maximum iterations reached.'
            self.result['is_successful'] = False

    def get_iteration_descriptions(self):
        descriptions = []
        iterations = self.result.get("iterations", {})
        iter_keys = sorted(iterations.keys(), key=lambda x: int(x) if isinstance(x, str) and x.isdigit() else x)
        for i, key in enumerate(iter_keys):
            it = iterations[key]
            a = it["a"]
            b = it["b"]
            s = it["s"]
            fa = it["f(a)"]
            fb = it["f(b)"]
            fs = it["f(s)"]
            error = it["error"]
            if i + 1 < len(iter_keys):
                next_it = iterations[iter_keys[i+1]]
                next_a = next_it["a"]
                next_b = next_it["b"]
            else:
                next_a = a
                next_b = b
            desc = (
                f"Інтервал: [{a:.3f}; {b:.3f}]\n"
                f"Обчислюємо точку s:\n"
                f"s = {s:.3f}\n"
                f"f(a) = {fa:.5f}\n"
                f"f(b) = {fb:.5f}\n"
                f"f(s) = {fs:.5f}\n"
                f"Наступний інтервал: [{next_a:.3f}; {next_b:.3f}]\n"
                f"Похибка: {error:.5f}"
            )
            descriptions.append(desc)
        return descriptions

class SuccessiveParabolicInterpolationMethod(OptimizationMethodBase):
    def __init__(self):
        super().__init__()
        self.name = 'Метод збіжності дотичних'
        self.category = OneDimensionalOptimizationCategory()
        self.iteration_result_key = "x_new"

    def _execute(self, input: dict):
        iteration = 0
        a, b = input["interval"]
        fa, fb = input["function"]([a]), input["function"]([b])

        # Initialize a third point c as the midpoint
        c = (a + b) / 2
        fc = input["function"]([c])

        while True:
            # Calculate the coefficients of the interpolating parabola
            numerator = (b - a)**2 * (fb - fc) - (b - c)**2 * (fb - fa)
            denominator = (b - a) * (fb - fc) - (b - c) * (fb - fa)

            if denominator == 0:
                self.result["failure_reason"] = "Zero denominator in interpolation formula."
                self.result["is_successful"] = False
                break

            x_new = b - 0.5 * numerator / denominator
            fx_new = input["function"]([x_new])

            cur_error = abs(x_new - b)

            self.result["iterations"][iteration] = {
                "a": a,
                "b": b,
                "c": c,
                "x_new": x_new,
                "f(a)": fa,
                "f(b)": fb,
                "f(c)": fc,
                "f(x_new)": fx_new,
                "error": cur_error
            }

            iteration += 1

            if cur_error < input["error"]:
                self.result["is_successful"] = True
                self.result["optimum"] = {
                    "position": {"x": x_new},
                    "value": fx_new
                }
                break

            if iteration >= input["max_iterations"]:
                self.result["failure_reason"] = "Maximum iterations reached."
                self.result["is_successful"] = False
                break

            # Update points for next iteration
            a, b, c = b, c, x_new
            fa, fb, fc = fb, fc, fx_new
        
    def get_iteration_descriptions(self):
        descriptions = []
        iterations = self.result.get("iterations", {})
        iter_keys = sorted(iterations.keys(), key=lambda x: int(x) if isinstance(x, str) and x.isdigit() else x)
        for i, key in enumerate(iter_keys):
            it = iterations[key]
            a = it["a"]
            b = it["b"]
            c = it["c"]
            x_new = it["x_new"]
            fa = it["f(a)"]
            fb = it["f(b)"]
            fc = it["f(c)"]
            fx_new = it["f(x_new)"]
            error = it["error"]
            if i + 1 < len(iter_keys):
                next_it = iterations[iter_keys[i+1]]
                next_a = next_it["a"]
                next_b = next_it["b"]
            else:
                next_a = a
                next_b = b
            desc = (
                f"Інтервал: [{a:.3f}; {b:.3f}]\n"
                f"Точка c = {c:.3f}\n"
                f"Нова точка: x_new = {x_new:.3f}\n"
                f"Значення функції:\n"
                f"f(a) = {fa:.5f}\n"
                f"f(b) = {fb:.5f}\n"
                f"f(c) = {fc:.5f}\n"
                f"f(x_new) = {fx_new:.5f}\n"
                f"Наступний інтервал: [{next_a:.3f}; {next_b:.3f}]\n"
                f"Похибка: {error:.5f}"
            )
            descriptions.append(desc)
        return descriptions

class RidderMethod(OptimizationMethodBase):
    def __init__(self):
        super().__init__()
        self.name = 'Метод Ріддера'
        self.category = OneDimensionalOptimizationCategory()
        self.iteration_result_key = "x_new"

    def _execute(self, input: dict):
        iteration = 0
        a, b = input["interval"]
        fa = input["function"]([a])
        fb = input["function"]([b])

        if fa * fb >= 0:
            self.result["is_successful"] = False
            self.result["failure_reason"] = "Function values at the interval endpoints must have opposite signs."
            return

        while iteration < input["max_iterations"]:
            c = (a + b) / 2
            fc = input["function"]([c])

            if abs(fc) < input["error"]:
                self.result["is_successful"] = True
                self.result["optimum"] = {
                    "position": {"x": c},
                    "value": fc
                }
                break

            s = (fc**2 - fa * fb) ** 0.5

            if s == 0:
                self.result["is_successful"] = False
                self.result["failure_reason"] = "Division by zero encountered."
                break

            d = (c - a) * fc / s

            if (fa - fb) < 0:
                d = -d

            x_new = c + d
            fx_new = input["function"]([x_new])

            self.result["iterations"][iteration] = {
                "a": a,
                "b": b,
                "c": c,
                "x_new": x_new,
                "f(a)": fa,
                "f(b)": fb,
                "f(c)": fc,
                "f(x_new)": fx_new,
                "error": abs(b - a)
            }

            iteration += 1

            if abs(fx_new) < input["error"]:
                self.result["is_successful"] = True
                self.result["optimum"] = {
                    "position": {"x": x_new},
                    "value": fx_new
                }
                break

            if fa * fx_new < 0:
                b = x_new
                fb = fx_new
            else:
                a = x_new
                fa = fx_new

        if iteration >= input["max_iterations"]:
            self.result["is_successful"] = False
            self.result["failure_reason"] = "Maximum iterations reached."

    def get_iteration_descriptions(self):
        descriptions = []
        iterations = self.result.get("iterations", {})
        iter_keys = sorted(iterations.keys(), key=lambda x: int(x) if isinstance(x, str) and x.isdigit() else x)
        for i, key in enumerate(iter_keys):
            it = iterations[key]
            a = it["a"]
            b = it["b"]
            c = it["c"]
            x_new = it["x_new"]
            fa = it["f(a)"]
            fb = it["f(b)"]
            fc = it["f(c)"]
            fx_new = it["f(x_new)"]
            error = it["error"]
            desc = (
                f"Інтервал: [{a:.3f}; {b:.3f}]\n"
                f"Точка c: {c:.3f}\n"
                f"Нова точка: x_new = {x_new:.3f}\n"
                f"Значення функції:\n"
                f"f(a) = {fa:.5f}\n"
                f"f(b) = {fb:.5f}\n"
                f"f(c) = {fc:.5f}\n"
                f"f(x_new) = {fx_new:.5f}\n"
                f"Похибка: {error:.5f}"
            )
            descriptions.append(desc)
        return descriptions

class NelderMeadMethod(OptimizationMethodBase):
    def __init__(self):
        super().__init__()
        self.name = 'Метод Нелдера-Міда'
        self.category = OneDimensionalOptimizationCategory()
        self.iteration_result_key = "center"
        self.input_identifiers = ["alpha", "gamma", "rho", "sigma"]

    def _generate_starting_points(self, interval, n=3):
        a, b = interval
        step = (b - a) / (n - 1)
        return numpy.array([a + i * step for i in range(n)])

    def _execute(self, input: dict):
        iteration = 0
        points = self._generate_starting_points(input["interval"])
        alpha = input.get("alpha", 1.0)
        gamma = input.get("gamma", 2.0)
        rho = input.get("rho", 0.5)
        sigma = input.get("sigma", 0.5)
        tol = input["error"]
        max_iter = input["max_iterations"]
        factor = -1 if input.get("is_max", False) else 1

        # Початкові значення функції
        def f(x): return factor * input["function"]([x])

        while iteration < max_iter:
            points = sorted(points, key=f)
            best, worst, second_worst = points[0], points[-1], points[-2]

            centroid = numpy.mean(points[:-1])
            reflection = centroid + alpha * (centroid - worst)
            f_reflection = f(reflection)

            if f_reflection < f(points[0]):
                # Розширення
                expansion = centroid + gamma * (reflection - centroid)
                f_expansion = f(expansion)
                points[-1] = expansion if f_expansion < f_reflection else reflection
            elif f(points[0]) <= f_reflection < f(second_worst):
                # Приймаємо відбиту точку
                points[-1] = reflection
            else:
                # Спробуємо стискання
                contraction = centroid + rho * (worst - centroid)
                f_contraction = f(contraction)
                if f_contraction < f(worst):
                    points[-1] = contraction
                else:
                    # Стискаємо весь симплекс
                    best = points[0]
                    points = [best + sigma * (p - best) for p in points]

            self.result["iterations"][iteration] = {
                "points": [float(p) for p in points],
                "center": float(sum(points) / len(points)),
                "f_values": [float(input["function"]([p])) for p in points],
                "error": float(numpy.std([input["function"]([p]) for p in points]))
            }

            if numpy.std([input["function"]([p]) for p in points]) < tol:
                break

            iteration += 1

        best_point = points[0]
        self.result["is_successful"] = iteration < max_iter
        self.result["optimum"] = {
            "position": {"x": float(best_point)},
            "value": input["function"]([best_point])
        }

        if iteration >= max_iter:
            self.result["failure_reason"] = "Maximum iterations reached."

    def get_iteration_descriptions(self):
        descriptions = []
        iterations = self.result.get("iterations", {})
        iter_keys = sorted(iterations.keys(), key=lambda x: int(x) if isinstance(x, str) and x.isdigit() else x)
        
        for i, key in enumerate(iter_keys):
            it = iterations[key]
            points = it["points"]
            f_values = it["f_values"]
            error = it["error"]
            
            desc = (
                f"Точки симплексу:\n"
            )
            
            for j, (point, f_val) in enumerate(zip(points, f_values)):
                desc += f"x{j+1} = {point:.3f}, f(x{j+1}) = {f_val:.5f}\n"
            
            desc += f"Похибка: {error:.5f}"
            descriptions.append(desc)
            
        return descriptions

ONEDIMENSIONAL_METHODS_LIST = [
    GoldenRatioMethod, 
    DichotomyMethod, 
    TernarySearchMethod, 
    FibonacciMethod, 
    NewtonRaphsonMethod, 
    ExhaustiveSearchMethod, 
    PowellsUnidimensionalMethod, 
    CubicSearchMethod, 
    SecantMethod, 
    BrentsMethod, 
    SuccessiveParabolicInterpolationMethod, 
    RidderMethod, 
    NelderMeadMethod, 
]

class NewtonMethod(OptimizationMethodBase):
    def __init__(self):
        super().__init__()
        self.name = "Метод Ньютона"
        self.category = MultiDimensionalOptimizationCategory()

    def _execute(self, input: dict):
        iteration = 0
        tol = input["error"]
        max_iter = input["max_iterations"]
        dim = input["dimensionality"]
        is_max = input["is_max"]

        x = numpy.array(input["start_point"], dtype=float)

        while iteration < max_iter:
            grad = numpy.array(input["gradient"](x))
            hess = numpy.array(input["hessian"](x))

            if is_max:
                grad = -grad
                hess = -hess

            try:
                delta = numpy.linalg.solve(hess, grad)
            except numpy.linalg.LinAlgError:
                self.result["is_successful"] = False
                self.result["failure_reason"] = "Hessian is singular or not invertible"
                return

            x_new = x - delta
            cur_error = numpy.linalg.norm(delta)

            self.result["iterations"][iteration] = {
                "x": x.tolist(),
                "gradient": grad.tolist(),
                "hessian": hess.tolist(),
                "delta": delta.tolist(),
                "f(x)": input["function"](x.tolist()),
                "error": float(cur_error)
            }

            if cur_error < tol:
                self.result["is_successful"] = True
                self.result["optimum"] = {
                    "position": {f"x{i+1}": float(val) for i, val in enumerate(x_new)},
                    "value": input["function"](x_new.tolist())
                }
                return

            x = x_new
            iteration += 1

        self.result["is_successful"] = False
        self.result["failure_reason"] = "Maximum iterations reached."

class ModifiedNewtonMethod(OptimizationMethodBase):
    def __init__(self):
        super().__init__()
        self.name = "Модифікований метод Ньютона"
        self.category = MultiDimensionalOptimizationCategory()

    def _execute(self, input: dict):
        x = numpy.array(input["start_point"], dtype=float)
        tol = input["error"]
        max_iter = input["max_iterations"]
        is_max = input["is_max"]
        factor = -1 if is_max else 1

        for iteration in range(max_iter):
            grad = numpy.array(input["gradient"](x))
            hess = numpy.array(input["hessian"](x))

            if is_max:
                grad = -grad
                hess = -hess

            try:
                hess_inv = numpy.linalg.inv(hess)
            except numpy.linalg.LinAlgError:
                self.result["is_successful"] = False
                self.result["failure_reason"] = "Матриця Гессе вироджена або не обернена."
                return

            direction = -hess_inv.dot(grad)

            alpha = 1.0
            x_next = x + alpha * direction
            f_current = input["function"](x.tolist())
            f_next = input["function"](x_next.tolist())

            # Лінійний пошук з backtracking (зменшуємо альфа, якщо треба)
            while factor * f_next > factor * f_current and alpha > 1e-6:
                alpha /= 2
                x_next = x + alpha * direction
                f_next = input["function"](x_next.tolist())

            error = numpy.linalg.norm(x_next - x)

            self.result["iterations"][iteration] = {
                "x": x.tolist(),
                "gradient": grad.tolist(),
                "hessian": hess.tolist(),
                "hessian_inv": hess_inv.tolist(),
                "direction": direction.tolist(),
                "alpha": alpha,
                "f(x)": f_current,
                "f(x_next)": f_next,
                "error": error
            }

            if error < tol:
                self.result["is_successful"] = True
                self.result["optimum"] = {
                    "position": {f"x{i+1}": float(val) for i, val in enumerate(x_next)},
                    "value": f_next
                }
                return

            x = x_next

        self.result["is_successful"] = False
        self.result["failure_reason"] = "Досягнуто максимум ітерацій"

class SimplexSearchMethod(OptimizationMethodBase):
    def __init__(self):
        super().__init__()
        self.name = "Пошук по симплексу"
        self.category = MultiDimensionalOptimizationCategory()
        self.input_identifiers = ["alpha"]

    def _execute(self, input: dict):
        n = input["dimensionality"]
        max_iter = input["max_iterations"]
        tol = input["error"]
        is_max = input["is_max"]
        alpha = input.get("alpha", 1.0)
        start = numpy.array(input["start_point"], dtype=float)
        factor = -1 if is_max else 1  # для уніфікованого сортування

        # Обчислення дельт для побудови регулярного симплекса
        delta1 = alpha * ((n + 1) ** 0.5 + n - 1) / (n * 2 ** 0.5)
        delta2 = alpha * ((n + 1) ** 0.5 - 1) / (n * 2 ** 0.5)

        # Побудова симплекса
        simplex = [start]
        for i in range(n):
            vertex = start.copy()
            for j in range(n):
                vertex[j] += delta2 if i != j else delta1
            simplex.append(vertex)

        iteration = 0
        history = {}

        while iteration < max_iter:
            # Обчислюємо значення ЦФ
            f_values = [input["function"](p.tolist()) for p in simplex]
            simplex = [p for _, p in sorted(zip(f_values, simplex), key=lambda x: factor * x[0])]
            f_values.sort(key=lambda v: factor * v)

            best, worst = simplex[0], simplex[-1]
            f_best, f_worst = f_values[0], f_values[-1]

            # Центр ваги всіх точок, крім найгіршої
            centroid = sum(simplex[:-1]) / n

            # Відображення найгіршої точки
            x_reflected = centroid * 2 - worst
            f_reflected = input["function"](x_reflected.tolist())

            self.result["iterations"][iteration] = {
                "simplex": [p.tolist() for p in simplex],
                "centroid": centroid.tolist(),
                "x_reflected": x_reflected.tolist(),
                "f(best)": f_best,
                "f(worst)": f_worst,
                "f(reflected)": f_reflected
            }

            # Замінюємо найгіршу точку, якщо відображення дало кращу
            if factor * f_reflected < factor * f_worst:
                simplex[-1] = x_reflected
            else:
                # Інакше — редукція симплекса до найкращої точки
                simplex = [best + 0.5 * (p - best) for p in simplex]

            # Критерій зупинки: стандартне відхилення
            if numpy.std(f_values) < tol:
                break

            iteration += 1

        # Повертаємо найкращу точку
        final_best = simplex[0]
        self.result["is_successful"] = iteration < max_iter
        self.result["optimum"] = {
            "position": {f"x{i+1}": float(xi) for i, xi in enumerate(final_best)},
            "value": input["function"](final_best.tolist())
        }

        if not self.result["is_successful"]:
            self.result["failure_reason"] = "Maximum iterations reached."

class HookeJeevesMethod(OptimizationMethodBase):
    def __init__(self):
        super().__init__()
        self.name = "Метод Гука-Дживса"
        self.category = MultiDimensionalOptimizationCategory()
        self.input_identifiers = ["initial_step", "step_reduction"]

    def _execute(self, input: dict):
        f = input["function"]
        n = input["dimensionality"]
        is_max = input["is_max"]
        tol = input["error"]
        max_iter = input["max_iterations"]
        step = input.get("initial_step", 0.5)
        step_reduction = input.get("step_reduction", 0.5)

        factor = -1 if is_max else 1
        x_base = numpy.array(input["start_point"], dtype=float)
        iteration = 0

        def explore(x, step_size):
            """Пошук в околі точки x по координатах"""
            x_new = x.copy()
            for i in range(n):
                f_curr = f(x_new.tolist())
                x_plus = x_new.copy()
                x_plus[i] += step_size
                f_plus = f(x_plus.tolist())

                x_minus = x_new.copy()
                x_minus[i] -= step_size
                f_minus = f(x_minus.tolist())

                if factor * f_plus < factor * f_curr:
                    x_new = x_plus
                elif factor * f_minus < factor * f_curr:
                    x_new = x_minus
                # інакше координата не змінюється
            return x_new

        while step > tol and iteration < max_iter:
            x_explored = explore(x_base, step)
            f_base = f(x_base.tolist())
            f_explored = f(x_explored.tolist())

            self.result["iterations"][iteration] = {
                "x_base": x_base.tolist(),
                "x_explored": x_explored.tolist(),
                "f(x_base)": f_base,
                "f(x_explored)": f_explored,
                "step": step
            }

            if factor * f_explored < factor * f_base:
                # рух у напрямку від base до explored
                x_pattern = x_explored + (x_explored - x_base)
                f_pattern = f(x_pattern.tolist())
                if factor * f_pattern < factor * f_explored:
                    x_base = x_pattern
                else:
                    x_base = x_explored
            else:
                step *= step_reduction  # зменшуємо крок

            iteration += 1

        self.result["is_successful"] = iteration < max_iter
        self.result["optimum"] = {
            "position": {f"x{i+1}": float(v) for i, v in enumerate(x_base)},
            "value": f(x_base.tolist())
        }

        if not self.result["is_successful"]:
            self.result["failure_reason"] = "Maximum iterations reached or step below tolerance."

class CauchyGradientDescentMethod(OptimizationMethodBase):
    def __init__(self):
        super().__init__()
        self.name = "Метод Коші(градієнтого спуску)"
        self.category = MultiDimensionalOptimizationCategory()

    def _execute(self, input: dict):
        x = numpy.array(input["start_point"], dtype=float)
        tol = input["error"]
        max_iter = input["max_iterations"]
        is_max = input["is_max"]
        grad_func = input["gradient"]
        expression = input["expression"]
        var_names = input["variables"]
        factor = 1 if is_max else -1

        # Символи
        gamma = sympy.Symbol('gamma')
        symbols = sympy.symbols(var_names)
        expr_sympy = sympy.sympify(expression)

        for iteration in range(max_iter):
            grad_k = numpy.array(grad_func(x.tolist()))
            grad_norm = numpy.linalg.norm(grad_k)

            if grad_norm < tol:
                self.result["is_successful"] = True
                self.result["optimum"] = {
                    "position": {f"x{i+1}": float(v) for i, v in enumerate(x)},
                }
                self.result["optimum"]["value"] = input["function"](x.tolist())
                return

            # Створити символічний вираз для x_next = x - gamma * grad_k
            x_expr = [sympy.Float(val) - factor * gamma * sympy.Float(grad_k[i]) for i, val in enumerate(x)]

            # Побудова нового градієнта в точці x_expr
            gradient_exprs = [sympy.diff(expr_sympy, var).subs(dict(zip(symbols, x_expr))) for var in symbols]

            # Побудова скалярного добутку нової ∇f(x_{k+1}) та старої ∇f(x_k)
            dot_product_expr = sum(sympy.Float(grad_k[i]) * gradient_exprs[i] for i in range(len(x)))

            # Розв'язання рівняння: ∇f(x_{k+1}) ⋅ ∇f(x_k) = 0
            gamma_solutions = sympy.solve(dot_product_expr, gamma)

            if not gamma_solutions:
                self.result["is_successful"] = False
                self.result["failure_reason"] = "Не вдалося знайти корінь для γ."
                return

            gamma_val = float(gamma_solutions[0])
            x_next = x - factor * gamma_val * grad_k
            f_val = input["function"](x.tolist())
            f_next = input["function"](x_next.tolist())

            self.result["iterations"][iteration + 1] = {
                "x": x.tolist(),
                "gradient": grad_k.tolist(),
                "gamma": factor * gamma_val,
                "x_next": x_next.tolist(),
                "f(x)": f_val,
                "f(x_next)": f_next,
                "||grad||": grad_norm
            }

            x = x_next

        self.result["is_successful"] = False
        self.result["failure_reason"] = "Досягнуто максимум ітерацій"

class SteepestDescentMethod(OptimizationMethodBase):
    def __init__(self):
        super().__init__()
        self.name = "Метод найшвидшого спуску"
        self.category = MultiDimensionalOptimizationCategory()

    def _execute(self, input: dict):
        f = input["function"]
        grad_f = input["gradient"]
        x = input["start_point"][:]
        tol = input["error"]
        max_iter = input["max_iterations"]
        is_max = input["is_max"]
        factor = -1 if is_max else 1

        def norm(v):
            return sum(a * a for a in v) ** 0.5

        def subtract(x1, x2):
            return [a - b for a, b in zip(x1, x2)]

        def multiply(vec, scalar):
            return [a * scalar for a in vec]

        def f_alpha(alpha, x, direction):
            point = subtract(x, multiply(direction, alpha))
            return f(point)

        for iteration in range(max_iter):
            grad = grad_f(x)
            grad_norm = norm(grad)

            if grad_norm < tol:
                self.result["is_successful"] = True
                self.result["optimum"] = {
                    "position": {f"x{i+1}": float(x[i]) for i in range(len(x))}
                }
                self.result["optimum"]["value"] = f(x)
                return

            direction = [factor * g for g in grad]  # напрямок спуску або підйому

            # Простий метод золотого перерізу для знаходження alpha
            a, b = 0, 1
            phi = (1 + 5 ** 0.5) / 2
            eps = 1e-5

            c = b - (b - a) / phi
            d = a + (b - a) / phi

            while abs(b - a) > eps:
                fc = f_alpha(c, x, direction)
                fd = f_alpha(d, x, direction)
                if fc < fd:
                    b = d
                else:
                    a = c
                c = b - (b - a) / phi
                d = a + (b - a) / phi

            alpha = (b + a) / 2

            x_next = subtract(x, multiply(direction, alpha))
            f_val = f(x)
            f_next = f(x_next)

            self.result["iterations"][iteration] = {
                "x": x,
                "gradient": grad,
                "||grad||": grad_norm,
                "alpha": alpha,
                "x_next": x_next,
                "f(x)": f_val,
                "f(x_next)": f_next
            }

            x = x_next

        self.result["is_successful"] = False
        self.result["failure_reason"] = "Досягнуто максимум ітерацій"

class FletcherReevesNonlinearConjugateGradientMethod(OptimizationMethodBase):
    def __init__(self):
        super().__init__()
        self.name = "Метод спряжених градієнтів (варіація Флетчера-Рівса)"
        self.category = MultiDimensionalOptimizationCategory()

    def _execute(self, input: dict):
        iteration = 0
        tol = input["error"]
        max_iter = input["max_iterations"]
        is_max = input["is_max"]

        x = numpy.array(input["start_point"], dtype=float)
        grad = numpy.array(input["gradient"](x.tolist()))
        if is_max:
            grad = -grad

        d = -grad  # Напрям спуску

        while iteration < max_iter:
            grad = numpy.array(input["gradient"](x.tolist()))
            if is_max:
                grad = -grad

            # Пошук оптимального кроку alpha вздовж напрямку d (лінійна мінімізація)
            def phi(alpha):
                point = x + alpha * d
                return -input["function"](point.tolist()) if is_max else input["function"](point.tolist())

            alpha = self._line_search(phi)

            x_new = x + alpha * d
            grad_new = numpy.array(input["gradient"](x_new.tolist()))
            if is_max:
                grad_new = -grad_new

            beta = numpy.dot(grad_new, grad_new) / (numpy.dot(grad, grad) + 1e-12)
            d = -grad_new + beta * d
            cur_error = numpy.linalg.norm(x_new - x)

            self.result["iterations"][iteration] = {
                "x": x.tolist(),
                "gradient": grad.tolist(),
                "direction": d.tolist(),
                "alpha": alpha,
                "f(x)": input["function"](x.tolist()),
                "error": float(cur_error)
            }

            if cur_error < tol:
                self.result["is_successful"] = True
                self.result["optimum"] = {
                    "position": {f"x{i+1}": float(val) for i, val in enumerate(x_new)},
                    "value": input["function"](x_new.tolist())
                }
                return

            x = x_new
            grad = grad_new
            iteration += 1

        self.result["is_successful"] = False
        self.result["failure_reason"] = "Maximum iterations reached."

    def _line_search(self, phi, alpha_init=1.0, tau=0.5, c=1e-4, max_iter=20):
        """
        Найпростіший варіант лінійного пошуку — метод Арміхо (backtracking line search)
        """
        alpha = alpha_init
        phi0 = phi(0.0)
        dphi0 = (phi(1e-8) - phi0) / 1e-8  # чисельна похідна

        for _ in range(max_iter):
            if phi(alpha) <= phi0 + c * alpha * dphi0:
                return alpha
            alpha *= tau
        return alpha

class HestenesStiefelNonlinearConjugateGradientMethod(OptimizationMethodBase):
    def __init__(self):
        super().__init__()
        self.name = "Метод спряжених градієнтів (варіація Гестенеса-Штіфеля)"
        self.category = MultiDimensionalOptimizationCategory()

    def _execute(self, input: dict):
        x = numpy.array(input["start_point"], dtype=float)
        tol = input["error"]
        max_iter = input["max_iterations"]
        is_max = input.get("is_max", False)

        # Цільова функція та градієнт (з урахуванням мін/макс)
        factor = -1 if is_max else 1
        f = lambda x: factor * input["function"](x.tolist())
        grad_f = lambda x: factor * numpy.array(input["gradient"](x.tolist()))

        g = grad_f(x)
        d = -g
        iteration = 0

        while iteration < max_iter:
            # Лінійний пошук (backtracking Armijo)
            def phi(alpha):
                return f(x + alpha * d)
            alpha = self._line_search(phi)

            x_new = x + alpha * d
            g_new = grad_f(x_new)

            delta_g = g_new - g
            denominator = numpy.dot(delta_g, d)

            if abs(denominator) < 1e-12:
                beta = 0
            else:
                beta = numpy.dot(g_new, delta_g) / denominator

            d = -g_new + beta * d
            cur_error = numpy.linalg.norm(x_new - x)

            self.result["iterations"][iteration] = {
                "x": x.tolist(),
                "gradient": g.tolist(),
                "direction": d.tolist(),
                "alpha": float(alpha),
                "f(x)": input["function"](x.tolist()),
                "error": float(cur_error)
            }

            if cur_error < tol:
                self.result["is_successful"] = True
                self.result["optimum"] = {
                    "position": {f"x{i+1}": float(val) for i, val in enumerate(x_new)},
                    "value": input["function"](x_new.tolist())
                }
                return

            x = x_new
            g = g_new
            iteration += 1

        self.result["is_successful"] = False
        self.result["failure_reason"] = "Maximum iterations reached."

    def _line_search(self, phi, alpha_init=1.0, tau=0.5, c=1e-4, max_iter=20):
        """Backtracking line search за правилом Арміхо"""
        alpha = alpha_init
        phi0 = phi(0)
        dphi0 = (phi(1e-8) - phi0) / 1e-8

        for _ in range(max_iter):
            if phi(alpha) <= phi0 + c * alpha * dphi0:
                return alpha
            alpha *= tau
        return alpha

class DaiYuanConjugateGradientMethod(OptimizationMethodBase):
    def __init__(self):
        super().__init__()
        self.name = "Метод спряжених градієнтів (варіація Дая-Юаня)"
        self.category = MultiDimensionalOptimizationCategory()

    def _execute(self, input: dict):
        iteration = 0
        tol = input["error"]
        max_iter = input["max_iterations"]
        is_max = input["is_max"]

        x = numpy.array(input["start_point"], dtype=float)
        grad = numpy.array(input["gradient"](x.tolist()))
        if is_max:
            grad = -grad

        d = -grad  # напрямок спуску

        while iteration < max_iter:
            # Функція для лінійного пошуку вздовж напрямку d
            def phi(alpha):
                x_trial = x + alpha * d
                return -input["function"](x_trial.tolist()) if is_max else input["function"](x_trial.tolist())

            alpha = self._line_search(phi)
            x_new = x + alpha * d

            grad_new = numpy.array(input["gradient"](x_new.tolist()))
            if is_max:
                grad_new = -grad_new

            y = grad_new - grad
            denom = numpy.dot(d, y)
            if abs(denom) < 1e-12:
                self.result["is_successful"] = False
                self.result["failure_reason"] = "Division by near-zero in Dai–Yuan beta."
                return

            beta = numpy.dot(grad_new, grad_new) / denom
            d = -grad_new + beta * d
            cur_error = numpy.linalg.norm(x_new - x)

            self.result["iterations"][iteration] = {
                "x": x.tolist(),
                "gradient": grad.tolist(),
                "direction": d.tolist(),
                "alpha": alpha,
                "beta": beta,
                "f(x)": input["function"](x.tolist()),
                "error": float(cur_error)
            }

            if cur_error < tol:
                self.result["is_successful"] = True
                self.result["optimum"] = {
                    "position": {f"x{i+1}": float(val) for i, val in enumerate(x_new)},
                    "value": input["function"](x_new.tolist())
                }
                return

            x = x_new
            grad = grad_new
            iteration += 1

        self.result["is_successful"] = False
        self.result["failure_reason"] = "Maximum iterations reached."

    def _line_search(self, phi, alpha_init=1.0, tau=0.5, c=1e-4, max_iter=20):
        """Простий line search із правилом Арміхо"""
        alpha = alpha_init
        phi0 = phi(0.0)
        dphi0 = (phi(1e-8) - phi0) / 1e-8

        for _ in range(max_iter):
            if phi(alpha) <= phi0 + c * alpha * dphi0:
                return alpha
            alpha *= tau
        return alpha

class LevenbergMarquardtMethod(OptimizationMethodBase):
    def __init__(self):
        super().__init__()
        self.name = "Алгоритм Левенберга–Марквардта"
        self.category = MultiDimensionalOptimizationCategory()
        self.input_identifiers = ["lambda"]

    def _execute(self, input: dict):
        iteration = 0
        tol = input["error"]
        max_iter = input["max_iterations"]
        dim = input["dimensionality"]
        is_max = input["is_max"]
        lam = input["lambda"]

        x = numpy.array(input["start_point"], dtype=float)

        while iteration < max_iter:
            grad = numpy.array(input["gradient"](x.tolist()))
            hess = numpy.array(input["hessian"](x.tolist()))

            if is_max:
                grad = -grad
                hess = -hess

            if numpy.linalg.norm(grad) < tol:
                break

            identity = numpy.eye(dim)
            try:
                step = -numpy.linalg.solve(hess + lam * identity, grad)
            except numpy.linalg.LinAlgError:
                self.result["is_successful"] = False
                self.result["failure_reason"] = "Singular matrix during step calculation"
                return

            x_new = x + step
            f_x = input["function"](x.tolist())
            f_x_new = input["function"](x_new.tolist())

            self.result["iterations"][iteration] = {
                "x": x.tolist(),
                "gradient": grad.tolist(),
                "hessian": hess.tolist(),
                "lambda": lam,
                "step": step.tolist(),
                "f(x)": f_x,
                "f(x_new)": f_x_new,
                "error": float(numpy.linalg.norm(step))
            }

            if (is_max and f_x_new > f_x) or (not is_max and f_x_new < f_x):
                x = x_new
                lam *= 0.5
            else:
                lam *= 2.0

            if numpy.linalg.norm(step) < tol:
                break

            iteration += 1

        self.result["optimum"] = {
            "position": {f"x{i+1}": float(val) for i, val in enumerate(x)},
            "value": input["function"](x.tolist())
        }

        self.result["is_successful"] = iteration < max_iter
        if not self.result["is_successful"]:
            self.result["failure_reason"] = "Maximum iterations reached."

class DevidoneFletcherPowellMethod(OptimizationMethodBase):
    def __init__(self):
        super().__init__()
        self.name = "Метод Девідона–Флетчера–Пауела"
        self.category = MultiDimensionalOptimizationCategory()

    def _execute(self, input: dict):
        iteration = 0
        tol = input["error"]
        max_iter = input["max_iterations"]
        dim = input["dimensionality"]
        is_max = input["is_max"]

        x = numpy.array(input["start_point"], dtype=float)
        grad = numpy.array(input["gradient"](x.tolist()))
        if is_max:
            grad = -grad

        H = numpy.eye(dim)  # Початкове наближення до оберненого гессіана

        while iteration < max_iter:
            d = -H @ grad  # Напрямок спуску

            def phi(alpha):
                point = x + alpha * d
                f_val = input["function"](point.tolist())
                return -f_val if is_max else f_val

            alpha = self._golden_section_search(phi)

            x_new = x + alpha * d
            grad_new = numpy.array(input["gradient"](x_new.tolist()))
            if is_max:
                grad_new = -grad_new

            s = x_new - x
            y = grad_new - grad

            s = s.reshape(-1, 1)
            y = y.reshape(-1, 1)
            Hs = H @ y

            numerator1 = s @ s.T
            denominator1 = float(s.T @ y) + 1e-12
            numerator2 = Hs @ Hs.T
            denominator2 = float(y.T @ Hs) + 1e-12

            H = H + numerator1 / denominator1 - numerator2 / denominator2

            cur_error = numpy.linalg.norm(x_new - x)

            self.result["iterations"][iteration + 1] = {
                "x": x.tolist(),
                "direction": d.tolist(),
                "alpha": alpha,
                "gradient": grad.tolist(),
                "H_inv": H.tolist(),
                "f(x)": input["function"](x.tolist()),
                "error": float(cur_error)
            }

            if cur_error < tol:
                break

            x = x_new
            grad = grad_new
            iteration += 1

        self.result["optimum"] = {
            "position": {f"x{i+1}": float(val) for i, val in enumerate(x)},
            "value": input["function"](x.tolist())
        }
        self.result["is_successful"] = iteration < max_iter
        if not self.result["is_successful"]:
            self.result["failure_reason"] = "Maximum iterations reached."

    def _golden_section_search(self, phi, a=0, b=1, tol=1e-5, max_iter=100):
        """
        Лінійний пошук методом золотого перерізу на відрізку [a, b]
        """
        gr = (5 ** 0.5 - 1) / 2  # ≈ 0.618
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = phi(c)
        fd = phi(d)

        for _ in range(max_iter):
            if abs(b - a) < tol:
                return (b + a) / 2
            if fc < fd:
                b, fd = d, fc
                d = c
                c = b - gr * (b - a)
                fc = phi(c)
            else:
                a, fc = c, fd
                c = d
                d = a + gr * (b - a)
                fd = phi(d)

        return (b + a) / 2

class BroydenFletcherGoldfarbShannoMethod(OptimizationMethodBase):
    def __init__(self):
        super().__init__()
        self.name = "Метод Бройдена-Флетчера-Гольдфарба-Шенно"
        self.category = MultiDimensionalOptimizationCategory()

    def _execute(self, input: dict):
        iteration = 0
        tol = input["error"]
        max_iter = input["max_iterations"]
        dim = input["dimensionality"]
        is_max = input["is_max"]

        x = numpy.array(input["start_point"], dtype=float)
        H = numpy.eye(dim)  # Початкове наближення до зворотного гессіана
        grad = numpy.array(input["gradient"](x.tolist()))
        if is_max:
            grad = -grad

        while iteration < max_iter:
            if numpy.linalg.norm(grad) < tol:
                break

            direction = -numpy.dot(H, grad)

            # Лінійний пошук: знайти оптимальний крок α
            def phi(alpha):
                point = x + alpha * direction
                return -input["function"](point.tolist()) if is_max else input["function"](point.tolist())

            alpha = self._golden_section_search(phi)

            s = alpha * direction
            x_new = x + s
            grad_new = numpy.array(input["gradient"](x_new.tolist()))
            if is_max:
                grad_new = -grad_new

            y = grad_new - grad

            ys = numpy.dot(y, s)
            if ys > 1e-10:
                rho = 1.0 / ys
                I = numpy.eye(dim)
                outer_sy = numpy.outer(s, y)
                outer_ys = numpy.outer(y, s)
                outer_ss = numpy.outer(s, s)

                H = (I - rho * outer_sy) @ H @ (I - rho * outer_ys) + rho * outer_ss

            self.result["iterations"][iteration] = {
                "x": x.tolist(),
                "gradient": grad.tolist(),
                "direction": direction.tolist(),
                "alpha": alpha,
                "step": s.tolist(),
                "f(x)": input["function"](x.tolist()),
                "error": float(numpy.linalg.norm(s))
            }

            x = x_new
            grad = grad_new
            iteration += 1

        self.result["optimum"] = {
            "position": {f"x{i+1}": float(val) for i, val in enumerate(x)},
            "value": input["function"](x.tolist())
        }
        self.result["is_successful"] = iteration < max_iter
        if not self.result["is_successful"]:
            self.result["failure_reason"] = "Maximum iterations reached."

    def _golden_section_search(self, phi, a=0, b=1, tol=1e-5, max_iter=100):
        """
        Лінійний пошук методом золотого перерізу на відрізку [a, b]
        """
        gr = (5 ** 0.5 - 1) / 2  # ≈ 0.618
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = phi(c)
        fd = phi(d)

        for _ in range(max_iter):
            if abs(b - a) < tol:
                return (b + a) / 2
            if fc < fd:
                b, fd = d, fc
                d = c
                c = b - gr * (b - a)
                fc = phi(c)
            else:
                a, fc = c, fd
                c = d
                d = a + gr * (b - a)
                fd = phi(d)

        return (b + a) / 2

class BroydenMethod(OptimizationMethodBase):
    def __init__(self):
        super().__init__()
        self.name = "Метод Бройдена"
        self.category = MultiDimensionalOptimizationCategory()

    def _execute(self, input: dict):
        iteration = 0
        tol = input["error"]
        max_iter = input["max_iterations"]
        dim = input["dimensionality"]
        is_max = input["is_max"]

        x = numpy.array(input["start_point"], dtype=float)
        grad = numpy.array(input["gradient"](x.tolist()))
        if is_max:
            grad = -grad

        eta = numpy.eye(dim)  # Початкове наближення оберненого гессіана

        while iteration < max_iter:
            direction = -eta.dot(grad)
            x_new = x + direction
            grad_new = numpy.array(input["gradient"](x_new.tolist()))
            if is_max:
                grad_new = -grad_new

            s = x_new - x
            y = grad_new - grad

            sy = numpy.dot(s, y)
            if abs(sy) < 1e-8:
                # Пропускаємо оновлення η, але не перериваємо метод
                eta_update_successful = False
            else:
                # Формула (8.6): η_{k+1} = η_k + ((s - η_k y) s^T) / (s^T y)
                eta += numpy.outer(s - eta @ y, s) / sy
                eta_update_successful = True

            f_x = input["function"](x.tolist())
            f_x_new = input["function"](x_new.tolist())

            self.result["iterations"][iteration] = {
                "x": x.tolist(),
                "f(x)": f_x,
                "gradient": grad.tolist(),
                "step": direction.tolist(),
                "eta_update": eta_update_successful,
                "error": float(numpy.linalg.norm(direction))
            }

            if numpy.linalg.norm(direction) < tol:
                break

            x = x_new
            grad = grad_new
            iteration += 1

        self.result["optimum"] = {
            "position": {f"x{i+1}": float(v) for i, v in enumerate(x)},
            "value": input["function"](x.tolist())
        }

        self.result["is_successful"] = iteration < max_iter
        if not self.result["is_successful"]:
            self.result["failure_reason"] = "Maximum iterations reached."

class NelderMeadMultidimensionalMethod(OptimizationMethodBase):
    def __init__(self):
        super().__init__()
        self.name = "Метод Нелдера-Міда"
        self.category = MultiDimensionalOptimizationCategory()

    def _execute(self, input: dict):
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink

        dim = input["dimensionality"]
        is_max = input["is_max"]
        tol = input["error"]
        max_iter = input["max_iterations"]
        start = numpy.array(input["start_point"], dtype=float)

        # Створюємо симплекс з n+1 точок
        simplex = [start.copy()]
        for i in range(dim):
            point = start.copy()
            if point[i] != 0.0:
                point[i] += 0.05 * abs(point[i])
            else:
                point[i] = 0.05
            simplex.append(point)
        simplex = numpy.array(simplex)

        def f(x): return input["function"](x.tolist())

        # Інвертуючи сортування, забезпечимо однакову логіку для max/min
        factor = -1 if is_max else 1

        iteration = 0
        while iteration < max_iter:
            # Сортуємо симплекс від найкращої до найгіршої точки
            simplex = sorted(simplex, key=lambda x: factor * f(x))
            f_values = [f(p) for p in simplex]

            best = simplex[0]
            worst = simplex[-1]
            second_worst = simplex[-2]

            # Центроїд без найгіршої точки
            centroid = numpy.mean(simplex[:-1], axis=0)

            # Відбиття
            xr = centroid + alpha * (centroid - worst)
            f_xr = f(xr)

            self.result["iterations"][iteration] = {
                "simplex": [p.tolist() for p in simplex],
                "centroid": centroid.tolist(),
                "xr": xr.tolist(),
                "f(xr)": f_xr
            }

            if factor * f_xr < factor * f_values[0]:  # Краща точка → Розширення
                xe = centroid + gamma * (xr - centroid)
                f_xe = f(xe)
                if factor * f_xe < factor * f_xr:
                    simplex[-1] = xe
                else:
                    simplex[-1] = xr
            elif factor * f_values[0] <= factor * f_xr < factor * f_values[-2]:  # Прийнятна точка
                simplex[-1] = xr
            else:
                # Контракція
                xc = centroid + rho * (worst - centroid)
                f_xc = f(xc)
                if factor * f_xc < factor * f_values[-1]:
                    simplex[-1] = xc
                else:
                    # Стискання всього симплексу до найкращої точки
                    best = simplex[0]
                    simplex = [best + sigma * (p - best) for p in simplex]

            # Критерій зупинки — стандартне відхилення значень функції
            std_dev = numpy.std([f(p) for p in simplex])
            self.result["iterations"][iteration]["σ"] = std_dev
            if std_dev < tol:
                break

            iteration += 1

        best_point = simplex[0]
        self.result["is_successful"] = True
        self.result["optimum"] = {
            "position": {f"x{i+1}": float(xi) for i, xi in enumerate(best_point)},
            "value": f(best_point)
        }

        if iteration >= max_iter:
            self.result["is_successful"] = False
            self.result["failure_reason"] = "Maximum iterations reached"

class ParticleSwarmMethod(OptimizationMethodBase):
    def __init__(self):
        super().__init__()
        self.name = "Метод рою часток"
        self.category = MultiDimensionalOptimizationCategory()
        self.input_identifiers = ["bounds", "population_size"]

    def _execute(self, input: dict):
        import numpy as np

        dim = input["dimensionality"]
        n_particles = input.get("population_size", 30)
        max_iter = input["max_iterations"]
        tol = input["error"]
        is_max = input["is_max"]
        bounds = input["bounds"]
        func = input["function"]

        # PSO parameters
        w = 0.7       # inertia
        c1 = 1.5      # cognitive
        c2 = 1.5      # social

        # Initialization
        particles = np.random.rand(n_particles, dim)
        for d in range(dim):
            min_b, max_b = bounds[d]
            particles[:, d] = min_b + particles[:, d] * (max_b - min_b)

        velocities = np.random.randn(n_particles, dim) * 0.1
        personal_best_pos = particles.copy()
        personal_best_val = np.array([func(p.tolist()) for p in personal_best_pos])
        if not is_max:
            personal_best_val *= -1

        global_best_index = np.argmax(personal_best_val)
        global_best_pos = personal_best_pos[global_best_index]
        global_best_val = personal_best_val[global_best_index]

        for iteration in range(max_iter):
            for i in range(n_particles):
                r1, r2 = np.random.rand(dim), np.random.rand(dim)
                velocities[i] = (
                    w * velocities[i]
                    + c1 * r1 * (personal_best_pos[i] - particles[i])
                    + c2 * r2 * (global_best_pos - particles[i])
                )
                particles[i] += velocities[i]

                # Apply bounds
                for d in range(dim):
                    particles[i, d] = np.clip(particles[i, d], *bounds[d])

                val = func(particles[i].tolist())
                val = val if is_max else -val

                if val > personal_best_val[i]:
                    personal_best_val[i] = val
                    personal_best_pos[i] = particles[i].copy()

                    if val > global_best_val:
                        global_best_val = val
                        global_best_pos = particles[i].copy()

            self.result["iterations"][iteration] = {
                "best_value": float(global_best_val if is_max else -global_best_val),
                "best_position": global_best_pos.tolist()
            }

            if np.std(personal_best_val) < tol:
                break

        final_value = global_best_val if is_max else -global_best_val
        self.result["optimum"] = {
            "position": {f"x{i+1}": float(x) for i, x in enumerate(global_best_pos)},
            "value": float(final_value)
        }
        self.result["is_successful"] = True

class DifferentialEvolutionMethod(OptimizationMethodBase):
    def __init__(self):
        super().__init__()
        self.name = "Метод диференціальної еволюції"
        self.category = MultiDimensionalOptimizationCategory()
        self.input_identifiers = ["bounds", "population_size", "F", "CR"]

    def _execute(self, input: dict):
        dim = input["dimensionality"]
        bounds = input["bounds"]
        func = input["function"]
        is_max = input["is_max"]
        pop_size = input.get("population_size", 10 * dim)
        F = input.get("F", 0.8)
        CR = input.get("CR", 0.9)
        tol = input["error"]
        max_iter = input["max_iterations"]

        def evaluate(x):
            return func(x) if not is_max else -func(x)

        population = [
            numpy.array([numpy.random.uniform(low, high) for (low, high) in bounds])
            for _ in range(pop_size)
        ]
        scores = [evaluate(ind) for ind in population]
        best_idx = numpy.argmin(scores)
        best = population[best_idx]
        best_score = scores[best_idx]

        for iteration in range(max_iter):
            for i in range(pop_size):
                indices = list(range(pop_size))
                indices.remove(i)
                idxs = numpy.random.choice(indices, 3, replace=False)
                a, b, c = population[idxs[0]], population[idxs[1]], population[idxs[2]]
                mutant = a + F * (b - c)

                # Ensure bounds
                mutant = numpy.clip(mutant, [b[0] for b in bounds], [b[1] for b in bounds])

                cross_points = numpy.random.rand(dim) < CR
                if not numpy.any(cross_points):
                    cross_points[numpy.random.randint(0, dim)] = True

                trial = numpy.where(cross_points, mutant, population[i])
                score = evaluate(trial)

                if score < scores[i]:
                    population[i] = trial
                    scores[i] = score
                    if score < best_score:
                        best_score = score
                        best = trial

            self.result["iterations"][iteration] = {
                "best_position": best.tolist(),
                "best_value": -best_score if is_max else best_score,
                "error": float(numpy.std(scores))
            }

            if numpy.std(scores) < tol:
                break

        self.result["optimum"] = {
            "position": {f"x{i+1}": float(val) for i, val in enumerate(best)},
            "value": -best_score if is_max else best_score
        }
        self.result["is_successful"] = True

class SimulatedAnnealingMethod(OptimizationMethodBase):
    def __init__(self):
        super().__init__()
        self.name = "Алгоритм імітації відпалу"
        self.category = MultiDimensionalOptimizationCategory()
        self.input_identifiers = ["bounds", "initial_temperature", "cooling_rate"]

    def _execute(self, input: dict):
        x = numpy.array(input["start_point"], dtype=float)
        bounds = input["bounds"]
        func = input["function"]
        T = input.get("initial_temperature", 100.0)
        cooling_rate = input.get("cooling_rate", 0.95)
        tol = input["error"]
        max_iter = input["max_iterations"]
        is_max = input["is_max"]

        def evaluate(x):
            return func(x) if not is_max else -func(x)

        best = x.copy()
        best_value = evaluate(x)
        current_value = best_value

        for iteration in range(max_iter):
            # Генеруємо сусідню точку
            candidate = x + numpy.random.uniform(-1, 1, size=x.shape) * T
            candidate = numpy.clip(candidate, [b[0] for b in bounds], [b[1] for b in bounds])

            candidate_value = evaluate(candidate)
            delta = candidate_value - current_value

            # Ймовірність прийняття гіршого рішення
            if delta < 0 or numpy.random.rand() < numpy.exp(-delta / T):
                x = candidate
                current_value = candidate_value

                if candidate_value < best_value:
                    best = candidate
                    best_value = candidate_value

            self.result["iterations"][iteration] = {
                "x": x.tolist(),
                "f(x)": func(x.tolist()),
                "T": T,
                "error": float(numpy.abs(delta))
            }

            # Умова зупинки
            if numpy.abs(delta) < tol:
                break

            T *= cooling_rate  # охолодження

        self.result["optimum"] = {
            "position": {f"x{i+1}": float(val) for i, val in enumerate(best)},
            "value": -best_value if is_max else best_value
        }
        self.result["is_successful"] = True

MULTIDIMENSIONAL_METHODS_LIST = [
    NewtonMethod,
    ModifiedNewtonMethod,
    SimplexSearchMethod,
    HookeJeevesMethod,
    CauchyGradientDescentMethod,
    SteepestDescentMethod,
    FletcherReevesNonlinearConjugateGradientMethod,
    HestenesStiefelNonlinearConjugateGradientMethod,
    DaiYuanConjugateGradientMethod,
    LevenbergMarquardtMethod,
    DevidoneFletcherPowellMethod,
    BroydenFletcherGoldfarbShannoMethod,
    BroydenMethod,
    NelderMeadMultidimensionalMethod,
    ParticleSwarmMethod,
    DifferentialEvolutionMethod,
    SimulatedAnnealingMethod
]

def get_iteration_descriptions(self):
    descriptions = []
    iterations = self.result.get("iterations", {})
    iter_keys = sorted(iterations.keys(), key=lambda x: int(x) if isinstance(x, str) and x.isdigit() else x)
    for i, key in enumerate(iter_keys):
        it = iterations[key]
        a = it["a"]
        b = it["b"]
        c = it["c"]
        d = it["d"]
        fc = it["f(c)"]
        fd = it["f(d)"]
        error = it["error"]
        if i + 1 < len(iter_keys):
            next_it = iterations[iter_keys[i+1]]
            next_a = next_it["a"]
            next_b = next_it["b"]
        else:
            next_a = a
            next_b = b
        desc = (
            f"Інтервал: [{a:.3f}; {b:.3f}]\n"
            f"Обчислюємо точки c і d:\n"
            f"c = {c:.3f}, d = {d:.3f}\n"
            f"f(c) = {fc:.5f}, f(d) = {fd:.5f}\n"
            f"Наступний інтервал: [{next_a:.3f}; {next_b:.3f}]\n"
            f"Похибка: {error:.5f}"
        )
        descriptions.append(desc)
    return descriptions