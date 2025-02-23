import random

class SimpleStochasticSearch:
    def __init__(self, function, x_bounds, y_bounds, step = 0.1, iterations = 1000):
        self.function = function
        self.x_low, self.x_high = x_bounds
        self.y_low, self.y_high = y_bounds
        self.step = step
        self.iterations = iterations

    def execute(self):
        x = random.uniform(self.x_low, self.x_high)
        y = random.uniform(self.y_low, self.y_high)
        best_x = x
        best_y = y
        best_result = self.function(x, y)
        for _ in range(self.iterations):
            x = random.uniform(self.x_low, self.x_high)
            y = random.uniform(self.y_low, self.y_high)
            result = self.function(x, y)
            if result < best_result:
                best_result = result
                best_x = x
                best_y = y

        print(f'completed in {self.iterations} iterations')
        return best_x, best_y, best_result