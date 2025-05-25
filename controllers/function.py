import sympy
from typing import List

def create_function(expression: str, variables: List[str] = ["x"]):
    symbols = sympy.symbols(variables)
    expr_sympy = sympy.sympify(expression)

    def evaluate(values: List[float]) -> float:
        if len(values) != len(symbols):
            raise ValueError("Number of values must match the number of variables")
        subs = {symbol: value for symbol, value in zip(symbols, values)}
        result = expr_sympy.subs(subs)
        return float(result.evalf())

    return evaluate

def get_derivative(expression: str, by: str = "x") -> str:
    x = sympy.Symbol(by)
    expr = sympy.sympify(expression.replace('^', '**'))
    derivative = sympy.diff(expr, x)
    return str(derivative)

def get_gradient(expression: str, variables: List[str]):
    gradient = [create_function(get_derivative(expression, variable), variables) for variable in variables]

    def evaluate(values: List[float]) -> List[float]:
        return [func(values) for func in gradient]

    return evaluate

def get_hessian(expression: str, variables: List[str]):
    hesse = [[create_function(get_derivative(get_derivative(expression, var1), var2), variables) for var2 in variables] for var1 in variables]

    def evaluate(values: List[float]) -> List[float]:
        return [[func(values) for func in arr] for arr in hesse]

    return evaluate