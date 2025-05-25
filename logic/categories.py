class BaseCategory:
    def __init__(self, name):
        self.name = name
        self.input_identifiers = []

class OneDimensionalOptimizationCategory(BaseCategory):
    def __init__(self):
        super().__init__("Одновимірна оптимізація")
        self.input_identifiers = ["function", "error", "max_iterations", "is_max", "interval"]

class MultiDimensionalOptimizationCategory(BaseCategory):
    def __init__(self):
        super().__init__("Багатовимірна оптимізація")
        self.input_identifiers = ["function", "dimensionality", "error", "max_iterations", "is_max", "start_point"]