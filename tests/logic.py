from logic.methods import *
from controllers.function import *
import time, numpy as np

def optimization_tests(type: str):
    if type == "1D":
        # Функції та їх відомі мінімуми
        test_cases = [
            {"function": "(x - 1.5)**2 + 1", "true_min": 1.5},
            {"function": "x**2 - 4*x + 4", "true_min": 2.0},
            {"function": "x**4 - 3*x**3 + 2", "true_min": 2.25},
            {"function": "1 + x - 2.5*x^2 + 0.25*x^4", "true_min": 2.12842},
        ]

        for idx, case in enumerate(test_cases, 1):
            function_str = case["function"]
            true_min = case["true_min"]
            print(f"\nТест-функція №{idx}: {function_str}")
            print(f"Очікуваний мінімум: {true_min}\n")

            inputs = {
                "expression": function_str,
                "function": create_function(function_str),
                "derivative1": create_function(get_derivative(function_str)),
                "derivative2": create_function(get_derivative(get_derivative(function_str))),
                "error": 0.001,
                "interval": [0, 4],
                "start_point": 0,
                "is_max": False,
                "max_iterations": 500,
                "n_steps": 10,
                "delta": 0.01,
            }

            results = {}
            start = time.time()

            for METHOD in ONEDIMENSIONAL_METHODS_LIST:
                method = METHOD()
                method.run(inputs)
                results[method.name] = method.result

            # Оцінювання результатів
            passed, failed = [], []

            for name, result in results.items():
                pos_dict = result["optimum"]["position"]
                if isinstance(pos_dict, dict) and pos_dict:
                    found = list(pos_dict.values())[0]
                    diff = abs(found - true_min)
                    if diff <= 1.0:
                        passed.append(name)
                    else:
                        failed.append((name, diff))
                else:
                    failed.append((name, "немає результату"))

            total = len(ONEDIMENSIONAL_METHODS_LIST)
            print(f"Пройшло: {len(passed)}/{total}")

            if failed:
                print("Методи, що не пройшли перевірку (різниця з мінімумом):")
                for method, delta in failed:
                    print(f"   {method}: {delta} {'різниці' if delta != 'немає результату' else ''}")

            print(f"Час виконання: {round(time.time() - start, 4)} с")

    elif type == "ND":
        # Тестові функції з відомим мінімумом (x1, x2)
        test_cases = [
            {
                "function": "6*x1^2 - 8*x1*x2 + 3*x2^2 + x1 + 4*x2",
                "true_min": [-4.75, -7.0]
            },
            {
                "function": "4*x1^2 - 2*x1*x2 + 2*x2^2 - 12*x1 - 2*x2 + 4",
                "true_min": [1.8571, 1.4286]
            },
            {
                "function": "x1^2 + x2^2",
                "true_min": [0.0, 0.0]
            }
        ]

        for idx, case in enumerate(test_cases, 1):
            function_str = case["function"]
            true_min = np.array(case["true_min"])
            variables = ["x1", "x2"]

            print(f"\nТест-функція №{idx}: {function_str}")
            print(f"Очікуваний мінімум: {true_min.tolist()}\n")

            inputs = {
                "expression": function_str,
                "variables": variables,
                "function": create_function(function_str, variables),
                "gradient": get_gradient(function_str, variables),
                "hessian": get_hessian(function_str, variables),
                "error": 0.0001,
                "is_max": False,
                "max_iterations": 500,
                "dimensionality": 2,
                "start_point": [0, 0],
                "lambda": 100,
                "n_particles": 50,
                "bounds": [(-10, 10), (-10, 10)],
                "population_size": 30
            }

            results = {}
            start = time.time()

            for METHOD in MULTIDIMENSIONAL_METHODS_LIST:
                method = METHOD()
                method.run(inputs)
                results[method.name] = method.result

            # Аналіз результатів
            passed, failed = [], []

            for name, result in results.items():
                pos = result["optimum"]["position"]
                if len(pos) == 2:
                    diff = np.mean(np.abs(np.array([pos['x1'], pos['x2']]) - true_min))
                    if diff <= 1.0:
                        passed.append(name)
                    else:
                        failed.append((name, diff))
                else:
                    failed.append((name, "немає результату"))

            total = len(MULTIDIMENSIONAL_METHODS_LIST)
            print(f"Пройшло: {len(passed)}/{total}")

            if failed:
                print("Методи, що не пройшли перевірку (макс. різниця по координаті):")
                for method, delta in failed:
                    print(f"   {method} — різниця: {delta}")

            print(f"Час виконання: {round(time.time() - start, 4)} с")