import numpy as np

class Softmax():
    # comments
    def activate(self, output_values):
        exponentials = np.exp(input - np.max(input))
        return exponentials / exponentials.sum(axis = 0)
    
    def derivative(self, output):
        return 0

test = [3 , 4, 7, 1, 0.2, -3, -.5]
print(Softmax().activate(test))