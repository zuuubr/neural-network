import numpy
import scipy.special

class neural:
    def __init__(self, input, hidden, output, speed = 0.2):
        self.input = input
        self.hidden = hidden
        self.output = output
        self.speed = speed
        self.wih = numpy.random.normal(0.0, pow(self.hidden, -0.5), (self.hidden, self.input))
        self.who = numpy.random.normal(0.0, pow(self.output, -0.5), (self.output,self.hidden))
        self.activate = lambda x: scipy.special.expit(x)

    def train(self, input_list, target_list):
        inputs = numpy.array(input_list, ndmin = 2).T
        targets = numpy.array(target_list, ndmin = 2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activate(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activate(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        self.who += self.speed * numpy.dot((output_errors * final_outputs * (1 - final_outputs)), numpy.transpose(hidden_outputs))
        self.wih += self.speed * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), numpy.transpose(inputs))
        pass

    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin = 2).T

        hidden_input = numpy.dot(self.wih, inputs)
        hidden_output = self.activate(hidden_input)

        final_input = numpy.dot(self.who, hidden_output)
        final_output = self.activate(final_input)

        return final_output


input_neural = 784
hidden_neural = 100
output_neural = 10
speed = 0.05

a = neural(input_neural, hidden_neural, output_neural, speed)
training_file = open("../DataSet/mnist.txt", 'r')
training_file_list = training_file.readline()
training_file.close()

for record in training_file_list:
    all_values = training_file_list.split(',')
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = numpy.zeros(output_neural) + 0.01
    targets[int(all_values[0])] = 0.99
    a.train(inputs, targets)
    print(all_values[0])
    print(a.query((numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01), "\n")

    max = 0
    for i in range(1, 10):
        x = a.query((numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)[i]
        if x > max:
            index = i
            max = x

    print(index)
