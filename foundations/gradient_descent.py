class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        function_value = init
        for i in range(iterations):
            function_value = function_value - learning_rate * (2*function_value)
            if 2*function_value == 0:
                break
        return round(function_value,5)
