import numpy
import scipy.special
import matplotlib.pyplot


class NeuralNetwork:

	def __init__(self, inpnodes, hiddenodes, outputnodes, learning_rate):
		self.inodes = inpnodes
		self.hnodes = hiddenodes
		self.onodes = outputnodes
		self.lr = learning_rate
		self.wih = (numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes)))
		self.who = (numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes)))
		self.activation_function = lambda x: scipy.special.expit(x)

	def train(self, inputs_list, targets_list):
		inputs = numpy.array(inputs_list, ndmin = 2).T
		targets = numpy.array(targets_list, ndmin = 2).T

		hidden_inputs = numpy.dot(self.wih,inputs)
		hidden_outputs = self.activation_function(hidden_inputs)

		final_inputs = numpy.dot(self.who,hidden_outputs)
		final_outputs = self.activation_function(final_inputs)

		output_errors = targets - final_outputs
		hidden_errors = numpy.dot(self.who.T,output_errors)

		self.who += self.lr * numpy.dot((output_errors*final_outputs*(1 - final_outputs)),numpy.transpose(hidden_outputs))
		self.wih += self.lr * numpy.dot((hidden_errors*hidden_outputs*(1 - hidden_outputs)),numpy.transpose(inputs))
		pass

	def query(self, inputs_list):
		inputs = numpy.array(inputs_list, ndmin = 2).T

		hidden_inputs = numpy.dot(self.wih,inputs)
		hidden_outputs = self.activation_function(hidden_inputs)

		final_inputs = numpy.dot(self.who,hidden_outputs)
		final_outputs = self.activation_function(final_inputs)
		
		return final_outputs



input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.2

nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

training_datafile = open("mnist_train_100.csv","r")

for record in training_datafile:
	all_values = record.split(',')
	inputs = (numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
	targets = numpy.zeros(output_nodes) + 0.01
	targets[int(all_values[0])] = 0.99
	nn.train(inputs, targets)

training_datafile.close()

testing_datafile = open("mnist_test_10.csv","r")

goodres = 0.0
badres = 0.0

for record in testing_datafile:
	all_values = record.split(',')
	correct_label = int(all_values[0])
	print "correct label is",correct_label
	inputs = (numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
	outputs = nn.query(inputs)
	label = numpy.argmax(outputs)
	print "network's answer is",label
	if(label == correct_label):
		goodres += 1
	else:
		badres += 1

testing_datafile.close()

print (goodres)/(goodres+badres)*100

#all_values = testing_data_list[0].split(',')
#inputs = (numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
#print nn.query(inputs)


#print targets
#image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
#matplotlib.pyplot.imshow(image_array, cmap = 'Greys', interpolation = 'None')
#matplotlib.pyplot.show()

#print len(data_list)
#print "\n\n"
#print data_list[0]


#print nn.query([0.4, 0.5 , 0.3])
