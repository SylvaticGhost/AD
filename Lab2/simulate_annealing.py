import random
import math

class SimulateAnnealing:
    def __init__(self, function, x_bounds, y_bounds, step = 0.1, iterations = 1000):
        self.function = function
        self.x_low, self.x_high = x_bounds
        self.y_low, self.y_high = y_bounds
        self.step = step
        self.iterations = iterations
        self.additional_iterations = 0

        self.temp_func = lambda t: 1 / t
        self.min_temp = 0.0001
        self.max_temp = 50
        self.v = 0.99

    def execute(self):
        x = random.uniform(self.x_low, self.x_high)
        y = random.uniform(self.y_low, self.y_high)
        best_x = x
        best_y = y
        best_result = self.function(x, y)
        T = self.max_temp
        i = 1

        while i < self.iterations and T > self.min_temp:
            T = self.temp_func(self.max_temp, i)
            x_new = self.get_next(x, i, T, self.x_low, self.x_high)
            y_new = self.get_next(y, i, T, self.y_low, self.y_high)
            result = self.function(x_new, y_new)
            if result < best_result:
                best_result = result
                best_x = x_new
                best_y = y_new
            else:
                if random.uniform(0, 1) < self.get_probability(result, best_result, T):
                    x = x_new
                    y = y_new
            i += 1

            if i % 10 == 0:
                print(f'iteration: {i},x={x},y={y} T: {T}, best_result: {best_result}')

        print(f'completed in {i} iterations, additional iterations: {self.additional_iterations}')
        return best_x, best_y, best_result


    def get_next(self, x, k, T, a, b):
        inside_iterations = 0
        while True:
            z = random.gauss(0, 1)
            E = random.uniform(0, 1)
            Xk = x + z * T * ((1 + k / T) ** (2 * E) - 1)
            if a <= Xk <= b:
                self.additional_iterations += inside_iterations
                return Xk
            inside_iterations += 1


    @staticmethod
    def get_probability(result, best_result, T):
        return 1 if result < best_result else math.exp((best_result - result) / T)