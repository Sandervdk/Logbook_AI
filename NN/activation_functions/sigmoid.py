from math import pow, e, exp

class Sigmoid():
    # comments
    def activate(self, output):
        outputs = output.get_vector()
        values = []

        for val in outputs:
            values.append(1 / (1 + pow(e, -1 * val)))

        output.set_vector(values)
        return output
    
    def derivative(self, output):
        outputs = output.get_vector()
        values = []

        for val in outputs:
            f = 1 / (1 + exp(val))
            df = f * (1 - f)
            values.append(df)

        output.set_vector(values)
        return output


## test code
# test = [-6, -4, -2, -1, 0, 1, 2, 4, 6]
# print(Sigmoid().derivative(test))