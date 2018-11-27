#D:\Python\367-64b\python.exe
import csv
import tensorflow as tf


import sys
def nn( LR = .002,maxEpoch = 1000,L1 = 1500,L2=1500,L3=1500 ):
	# Notes: Often times the inputs are called X and the outputs are called Y
	# Uncomment any print statements to see debugging info
	# move sys.exit() around to exit your program for debugging

	##################################################
	#Read in data from training.csv and separate out inputs and outputs

	#Health,Knife,Gun,Enemies,Attack,Run,Wander,Hide
	trainX = []
	trainY = []
	file = open('training.csv', 'r', encoding='utf-8')
	fileReader = csv.reader(file)
	dataAsList = list(fileReader) # convert filereader data to a list

	#print ( dataAsList )  # should be 8 list of lists,  Outer list: 18 rows,  each row is list of 18 records





	# get data in correct format, separate out the inputs and outputsinto separate lists
	# trainX will contain a list of lists.  17 items in outter list, 4 items in inner list with values of Health,Knife,Gun,Enemy
	# trainY will contain the  17 items of the labels.  That is values of the classification Attack,Run,Wander,Hide

	for line in dataAsList[1:]:       # for each record in list, start at item 1 instead of 0 beacause row - contains headers
		feature = []
		label = []
		for item in line[:4]:   #copy  each input property into a list
			feature.append(float(item))
		for item in line[4:]:
			label.append(float(item))
		trainX.append(feature)  # add the list of 4 items to the big list
		trainY.append(label)
	file.close()


	#print (trainX) # list of inputs
	#print (trainY) # list of outputs

	############################################
	# Read in data for test data
	# read and convert test data

	testX = []
	testY = []
	file = open('test.csv', 'r', encoding='utf-8')
	fileReader = csv.reader(file)
	fileReader = list(fileReader)
	for line in fileReader[1:]:
		feature = []
		label = []
		for item in line[:4]:
			feature.append(float(item))

		for item in line[4:]:
			label.append(float(item))

		testX.append(feature)
		testY.append(label)
	file.close()


	#print (testX) # list of inputs
	#print (testY) # list of outputs





	################################################
	# Set up Neural Network pieces:  settings, weights and biases

	learningRate = LR # sets how much aggressive change happens to the weights during each epoch
	trainingEpochs = maxEpoch  # number of epochs to train.... we could also quit at a threshold
	batchSize = len(trainY)  # batches are used to load data in chunks when it can't all fit into memory.  Our data easily fits so we set to num records
	displayStep = 100 # for printing status as it trains
	numberOfInputs = 4  #number of features,   Health,Knife,Gun,Enemies
	numberOfOutputs = 4 #numbers of classifications,  Attack,Run,Wander,Hide

	nodesInHiddenLayer1 = L1
	nodesInHiddenLayer2 = L2
	nodesInHiddenLayer3 = L3


	#setup tensors placeholders,  see  https://www.tensorflow.org/api_docs/python/tf/placeholder


	X = tf.placeholder(name = 'X', dtype = tf.float32, shape = [None, 4])
	Y = tf.placeholder(name = 'Y', dtype = tf.float32, shape = [None, 4])




	# set up random weights between  Input-L1, L1-L2, L2-L3, L3-output
	# By default, tf.random.normal uses a mean of 0 and Standard Deviation of 1, see https://www.tensorflow.org/api_docs/python/tf/random/normal
	weights = {
		'hiddenLayer1': tf.Variable(tf.random.normal([numberOfInputs, nodesInHiddenLayer1])),
		'hiddenLayer2': tf.Variable(tf.random.normal([nodesInHiddenLayer1, nodesInHiddenLayer2])),
		'hiddenLayer3': tf.Variable(tf.random.normal([nodesInHiddenLayer2, nodesInHiddenLayer3])),
		'outputLayer': tf.Variable(tf.random.normal([nodesInHiddenLayer3, numberOfOutputs]))
	}



	####Uncomment all below to see the initial weights
	#print(weights)  # would only print the shapes
	#init = tf.global_variables_initializer()
	#with tf.Session() as debugSess:
	#    debugSess.run(init)
	#    print(debugSess.run(weights))
	####


	# setup random biases, for L1,L2,L3,Out with random numbers
	biases = {
		'biasLayer1': tf.Variable(tf.random.normal([nodesInHiddenLayer1])),
		'biasLayer2': tf.Variable(tf.random.normal([nodesInHiddenLayer2])),
		'biasLayer3': tf.Variable(tf.random.normal([nodesInHiddenLayer3])),
		'outputLayer': tf.Variable(tf.random.normal([numberOfOutputs]))
	}


	####Uncomment all below to see the initial bias
	#init = tf.global_variables_initializer()
	#with tf.Session() as debugSess:
	#    debugSess.run(init)
	#    print(debugSess.run(biases))
	####


	############################################
	def ForwardPropagateModel( x ):
		# Forward propagate.  SUM{ Input*Wo + bias }
		_layer1 = tf.add(tf.matmul(x, weights['hiddenLayer1']), biases['biasLayer1'])
		_layer2 = tf.add(tf.matmul(_layer1, weights['hiddenLayer2']), biases['biasLayer2'])
		_layer3 = tf.add(tf.matmul(_layer2, weights['hiddenLayer3']), biases['biasLayer3'])
		_outputLayer =  tf.matmul(_layer3, weights['outputLayer']) + biases['outputLayer']
		return _outputLayer



	###########################################
	# Construct Model, this is the output created from forward propagation
	logits = ForwardPropagateModel(X)  #often in tensorflow logits are the outputs


	###########################################
	# Define loss/cost function
	# pass in the calculated outputs and expected outputs.  
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = Y))

	###########################################
	# define the optimizer, this will adjust the weights
	optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
	trainingOp = optimizer.minimize(loss)


	# note.... X and Y still do not have any data in them yet so you can't print/debug X and Y yet until the session starts... or anything that uses it


	#######################################
	## This runs the neural network
	# Initializing the variables
	zeroCostCount = 0
	solutionFound = 0
	init = tf.global_variables_initializer()
	with tf.Session() as trainingSession:
		trainingSession.run(init)
		epochCounter = 0
		for epoch in range(trainingEpochs):
			averageCost = 0
			totalBatch = 1  # our data is small so we don't need lots of batches
			for i in range(totalBatch):
				batchX, batchY = trainX, trainY

				_, c = trainingSession.run([trainingOp, loss], feed_dict={X: batchX, Y: batchY})
				epochCounter += 1
			#  Code to exit when solution found, 
				averageCost += c / totalBatch
				if averageCost == 0:
			 		zeroCostCount += 1
			 		if zeroCostCount > 20:
			 			solutionFound = 1
			 			break
				else:
			 		zeroCostCount = 0 	
			if solutionFound == 1:
				break
			#if epoch % displayStep == 0:
			#	print("Epoch:", '%04d' % (epoch + 1), "cost={:.15f}".format(averageCost))
		#print ("Training Finished")

		# test
		prediction = tf.nn.softmax(logits)
		correctPrediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y,1))

		#calculate accurace
		accuracy = tf.reduce_mean(tf.cast(correctPrediction,tf.float32))
		#print("Accuracy:", accuracy.eval({X: testX, Y: testY}))
		return accuracy.eval({X: testX, Y: testY}),epochCounter

#print(nn(.005,200,2000,2000,1500))