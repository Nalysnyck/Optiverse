from logic.methods import *
from controllers.function import *
import time

def pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + f"{key:<13}", end='')
        if isinstance(value, dict):
            print()
            print('\t' * indent + '{')
            pretty(value, indent + 1)
            print('\t' * indent + '}')
        else:
            print(': ' + str(value))

def optimization_tests(type: str):
    if type == "1D":
        # One-dimensional tests
        function_str = "1+x-2.5*x^2+0.25*x^4"

        inputs = {
            "expression": function_str,
            "function": create_function(function_str),
            "derivative1": create_function(get_derivative(function_str)),
            "derivative2": create_function(get_derivative(get_derivative(function_str))),
            "error": 0.001,
            "interval": [-1, 2],
            "start_point": 0,
            "is_max": True,
            "max_iterations": 500,

            "n_steps": 10, # exhastive search
            "delta": 0.01, # Powell's method
        }
        results = {}
        start = time.time()
        for METHOD in ONEDIMENSIONAL_METHODS_LIST:
            method = METHOD()
            method.run(inputs)
            results[method.name] = method.result
            print(f"{method.name}")
            print(f"\tx          : {results[method.name]['optimum']['position']}")
            print(f"\titerations : {len(results[method.name]['iterations'])}\n")
        print(f"Total time: {time.time() - start}")
        # pretty(results)
    
    elif type == "ND":
        # Multi-dimensional tests
        # function_str = "4*x1^2-2*x1*x2+2*x2^2-12*x1-2*x2+4"
        # function_str = "4*x1^2-2*x1*x2+3.5*x2^2-x1"
        function_str = "6*x1^2-8*x1*x2+3*x2^2+x1+4*x2"
        variables = ["x1", "x2"]

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
            print(f"{method.name}")
            print(f"\tx          : {results[method.name]['optimum']['position']}")
            print(f"\titerations : {len(results[method.name]['iterations'])}\n")
        print(f"Total time: {time.time() - start}")
        # pretty(results)
    
    else:
        raise Exception(f"No such type of tests: {type}")