class Relu():
    # comments
    def activate(self, output):
        outputs = output.get_vector()
        values = []

        for val in outputs:
            values.append(max(0, val))

        output.set_vector(values)
        return output
    
    def derivative(self, output):
        outputs = output.get_vector()
        values = []

        for val in outputs:
            if val <= 0:
                values.append(0)
            else:
                values.append(1)

        output.set_vector(values)
        return output