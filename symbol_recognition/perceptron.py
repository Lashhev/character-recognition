
from numpy import exp, array, dot, tanh, power, ndarray, uint8
from numpy.random import random, seed
import csv 
import yaml
import cv2
symbols_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                             'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
                             'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
                             'Y', 'Z','0', '1', '2','3','4', '5','6',
                             '7', '8','9']
class Perceptron():
    def __init__(self, weight_number:int):
        seed(1)
        self.synaptic_weights = 2 * random((weight_number, 1)) - 1

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def __back_propogation(self, training_set_inputs, training_set_outputs):
        output = self.think(training_set_inputs)
        error = training_set_outputs - output
        adjustment = dot(training_set_inputs.T, error*(self.__sigmoid_derivative(output)))
        self.synaptic_weights += adjustment

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations, mu, q):
        for iteration in range(0, number_of_training_iterations):
            # Pass the training set through our neural network (a single neuron).
            s = self.think(training_set_inputs)
            y = tanh(q*s)
            dy = q*(1 - power(y, 2))


            # Calculate the error (The difference between the desired output
            # and the predicted output).
            error = training_set_outputs - y

            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.
            # signal = self.__sigmoid_derivative(output)
            adjustment = 2*mu*(dot((error*dy).T ,training_set_inputs).T)

            # Adjust the weights.
            self.synaptic_weights += adjustment

    # The neural network thinks.
    def think(self, inputs):
        # Pass inputs through our neural network (our single neuron).
        return self.__sigmoid(dot(inputs, self.synaptic_weights))

    def save_weight(self, stream, tag):
        calc_weights = dict()
        calc_weights[tag] = self.synaptic_weights.ravel().tolist()
        
        yaml.dump(calc_weights, stream)

    def load_weights(self, filename, tag):
        config = yaml.load(open(filename))
        self.synaptic_weights = config[tag]

def load_train_set(filename):
    with open(filename, 'r') as train_set_file:
        train_set = list()
        csv_reader = csv.reader(train_set_file)
        for row in csv_reader:
            train_set.append(row)
    return train_set

def output_results_accord_to_symbol(train_set, symbol):
    symbols = [int(x[0]) for x in train_set]
    symbols = [ list(map(int, [x == ord(symbol)])) for x in symbols]
    return array(symbols)

if __name__ == "__main__":

    #Intialise a single neuron neural network.
    
    neural_network = Perceptron(64)
    train_set = load_train_set('/home/lashhev/Documents/classificator/database2/dataset2.csv')
    # training_set_inputs = array([list(map(int, x[1:])) for x in train_set])
    training_set_inputs =array([list(map(int, train_set[0][1:]))])
    for i in range(0, len(symbols_list)):
        symbol = training_set_inputs
        image = array(255*symbol.reshape(8,8),dtype=uint8)
        cv2.imshow('Symbol', image)
        neural_network.load_weights('/home/lashhev/Documents/classificator/weights2.yaml',symbols_list[i])
        print (neural_network.think(symbol))
        cv2.waitKey(0)