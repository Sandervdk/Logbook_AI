from random import randint
import itertools
class Matrix():
    matrix = [[]]
    curr_size = 0 # neurons of current layer
    prev_size = 0 # neurons of previous layer

    def __init__(self, length = 0, width = 0, matrix = []):
        if len(matrix) > 0:
            self.matrix = matrix
            self.prev_size = len(matrix)
            self.curr_size = len(matrix[0])
        else: 
            matrix = []
            for i in range(width):
                vector = []
                for j in range(length):
                    vector.append(1 - randint(0, 2000) / 1000)
                matrix.append(vector)

            self.matrix = matrix
            self.prev_size = length
            self.curr_size = width

    def get_matrix(self):
        return self.matrix

    def set_matrix(self, matrix):
        self.matrix = matrix
        self.prev_size = len(matrix)
        self.curr_size = len(matrix[0])

    def add(self, matrix):
        matrix = matrix.get_matrix()

        if len(matrix) == self.curr_size and len(matrix[0]) == self.prev_size:
            for i in range(len(matrix)):
                for j in range(len(matrix[0])):
                    self.matrix[i][j] = self.matrix[i][j] + matrix[i][j]
        else:
            print(len(matrix), self.curr_size, len(matrix[0]), self.prev_size)

    def multiply(self, value):
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix[0])):
                self.matrix[i][j] = self.matrix[i][j] * value

    def identity_matrix(self, size):
        matrix = [[0 for x in range(size)] for x in range(size)]

        for i in range(size):
            matrix[i][i] = 1

        return matrix;

    def dot_product(self, inputs, biases):
        total = []
        inputs = inputs.get_vector()
        biases = biases.get_vector()
        
        for i in range(self.prev_size):
            sum = 0
            for j in range(self.curr_size):   
                sum += inputs[i] * self.matrix[j][i]
            total.append(sum + biases[i])

        return Vector(vector = total)

    
    def transposed_dot(self, vector):
        total = []
        vector = vector.get_vector()
        
        for i in range(self.curr_size):
            sum = 0
            for j in range(self.prev_size):   
                sum += vector[j] * self.matrix[i][j]
            total.append(sum) 
        
        return Vector(vector = total)


    def __str__(self):
        return ''.join(str(self.matrix))


class Vector():
    vector = []

    def __init__(self, size = 0, random = False, vector = []):
        if len(vector) > 0:
            self.vector = vector
        elif not random:
            self.vector = [0 for i in range(size)]
        else:
            self.vector = [randint(0, 1000) / 1000 for i in range(size)]

    def get_vector(self):
        return self.vector

    def set_vector(self, vector):
        self.vector = vector

    def identity_matrix(self, size):
        matrix = [[0 for x in range(size)] for x in range(size)]

        for i in range(size):
            matrix[i][i] = 1

        return matrix;

    def add(self, vector):
        new = []
        vector = vector.get_vector()

        if (len(vector)) == len(self.vector):
            for i in range(len(self.vector)):
                self.vector[i] = self.vector[i] + vector[i]

    def subtract(self, vector):
        new = []
        vector = vector.get_vector()

        if (len(vector)) == len(self.vector):
            for i in range(len(self.vector)):
                new.append(self.vector[i] - vector[i])
        
        return Vector(vector = new)

    def multiply(self, vector):
        new = []
        vector = vector.get_vector()
        # print(vector)
        # print(self.vector)

        if (len(vector)) == len(self.vector):
            for i in range(len(self.vector)):
                new.append(self.vector[i] * vector[i])

        return Vector(vector = new)

    def multiply_scalar(self, scalar):
        for i in range(len(self.vector)):
            self.vector[i] = self.vector[i] * scalar

    def dot_product(self, weight):
        total = []
        weight = weight.get_vector()
        # bias = bias.get_vector()

        for i in range(len(self.vector)):
            total.append(self.vector[i] * weight[i])

        return Matrix(vector = total)

    def dot_list(self, list_values):
        matrix = [[0 for j in range(len(self.vector))] for i in range(len(list_values))]
        # print(self.vector) ## bestaat niet~!!!!??
        # print(list_values)
        for i in range(len(list_values)):
            for j in range(len(self.vector)):
                matrix[i][j] = self.vector[j] * list_values[i]

        # print(matrix)

        return Matrix(matrix = matrix)

    def __str__(self):
        return ''.join(str(self.vector))