from activation_functions import sigmoid, mse, relu
import matrix, random

class Neural_network():
    previous_layer_size = 0
    input_size = 0
    output_size = 0
    layers = []

    learning_rate = .01
    mse = mse.MSE()

    def __init__(self):
        pass

    def create_layer(self, size, activation_function):
        biases = matrix.Vector(size, False)

        if self.previous_layer_size == 0:
            weights = matrix.Matrix(size, size)
        else:
            weights = matrix.Matrix(size, self.previous_layer_size)

        self.previous_layer_size = size;
        self.layers.append({'inputs': [], 'weights': weights, 'biases': biases, 'activation': activation_function, 'output': []})

    def feed_forward(self, input, next_layer):
        next_layer['inputs'] = input
        # print('\n input')
        # print(input)

        # print('\n biases')
        # print(next_layer['biases'])

        # print('\n weights: ')
        # print(next_layer['weights'])
        dot_values = next_layer['weights'].dot_product(input, next_layer['biases'])

        # print('\n dot values:')
        # print(dot_values)
        next_layer['output'] = next_layer['activation'].activate(dot_values)

        return next_layer['output']

    def predict(self, input_values):
        result = input_values

        # print(self.layers)
        for layer in self.layers:
            result = self.feed_forward(result, layer)

        return result

    def fit(self, inputs , outcomes, learning_rate = 0.01) :
        # Predict stores al intermeadiate results in the layer.
        predictions = self.predict(inputs)
        # print('prediction:')
        # print(predictions)
        # targets y - final outputs
        error = outcomes.subtract(predictions)
  

        # reversed model approach
        for layer in reversed(self.layers):
            self.update_layer(layer, error, learning_rate)
   
            # calculate hidden erorrs
            error = layer['weights'].transposed_dot(error)

    def update_layer(self, layer, errors, learning_rate):
        # get differentiated outputs
        # print('\n Error:')
        # print(errors)

        differentiated_output = layer['activation'].derivative(layer['output'])
        # print('\n Differentiated: ')
        # print(differentiated_output)

        # setup weights
        error_output = errors.multiply(differentiated_output)
        # print('\n erorr output')
        # print(error_output)


        delta_weights = error_output.dot_list(layer['inputs'].get_vector())
        delta_weights.multiply(learning_rate)
        # print('\n weights:')
        # print(delta_weights)
        # print()

        # update weights
        layer['weights'].add(delta_weights)

        # setup biases
        error_output.multiply_scalar(learning_rate)

        # update biases
        layer['biases'].add(error_output)
        # print('\n biases')
        # print(layer['biases'])

    # only works for single value outputs
    def accuracy_score(self, values, outcomes):
        right_predictions = 0

        for i in range(len(values)):
            predict = self.predict(values[i]).get_vector()
            predicted_outcome = [0] * len(predict)
            predict_index = predict.index(max(predict))
            predicted_outcome[predict_index] = 1
            
            if list(predicted_outcome) == list(outcomes[i].get_vector()):
                right_predictions += 1

        return right_predictions / len(values)
        


# %%
# inputs = matrix.Vector(15, True)
# nn = Neural_network()
# nn.create_layer(3, sigmoid.Sigmoid())
# nn.create_layer(1, sigmoid.Sigmoid())


# training_data1 = matrix.Vector(vector = [5, -5, 2])
# training_data2 = matrix.Vector(vector = [-5, 5, 2])

# outcome1 = matrix.Vector(vector = [1])
# outcome2 = matrix.Vector(vector = [0])

# # Trains the NN with two possible outcomes
# for i in range(5): 
#     nn.fit(training_data1, outcome1, learning_rate = 0.05)
#     nn.fit(training_data2, outcome2, learning_rate = 0.05)

# test_1 = matrix.Vector(vector = [5, -5, 2]) # outcome = 1
# test_2 = matrix.Vector(vector = [-5, 5, 2]) # outcome = 0
# test_3 = matrix.Vector(vector = [4, 0, 1]) # outcome = 1
# test_4 = matrix.Vector(vector = [-1, 4, 3]) # outcome = 0

# print(nn.predict(test_1))
# print(nn.predict(test_2))
# print(nn.predict(test_3))
# print(nn.predict(test_4))

