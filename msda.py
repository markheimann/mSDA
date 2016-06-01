#Implement mDA (Chen et. al, 2012)
#Take in data and probability of corruption
#"Corrupt" data (but marginalize out the (expected) corruption) and learn a reconstruction specified by weights
import numpy as np
#from numpy import *
import numpy.matlib
import math, time


#Learn a deep representation of data by reconstructing "corrupted" input but marginalizing out corruption
#data format: features are rows, data points are columns
#NEW: <num_data x num_features>
#Can optionally pass in precomputed mapping to use to transform data
	#(e.g. if transforming test data with mapping learned from training data)
def mDA(data, prob_corruption=None, use_nonlinearity = True, mapping = None):
	if mapping is None:
		mapping = compute_reconstruction_mapping(data, prob_corruption)
	representation = np.dot(data, mapping) #no nonlinearity
	if use_nonlinearity:
		representation = np.tanh(representation) #inject nonlinearity
	return mapping, representation

#Compute the mapping that reconstructs corrupted (in expectation) features
def compute_reconstruction_mapping(data, prob_corruption):
	#typecast to correct datatype
	if not (np.issubdtype(data.dtype, np.float) or np.issubdtype(data.dtype, np.integer)):
		print "data type ", data.dtype
		data.dtype = "float64"
	num_features = data.shape[1]

	#Represents the probability that a given feature will be corrupted
	feature_corruption_probs = np.ones((num_features, 1))*(1-prob_corruption)
	#TODO could automatically check if last "feature" is all 1s (i.e. bias)
	#instead of requiring user to tell us
	bias = False
	try:
		if np.allclose(np.ones((num_features,1)),data[:,-1]):
			bias = True
	except Exception as e:
		raise ValueError(e)
	if bias: #last term is actually a bias term, not an actual feature
		feature_corruption_probs[-1] = 1 #don't corrupt the bias term ever
	scatter_matrix = np.dot(data.transpose(), data)
	Q = scatter_matrix*(np.dot(feature_corruption_probs, feature_corruption_probs.transpose()))
	Q[np.diag_indices_from(Q)] = feature_corruption_probs[:,0] * np.diag(scatter_matrix)
	P = scatter_matrix * numpy.matlib.repmat(feature_corruption_probs, 1, num_features)

	#solve equation of the form x = BA^-1, or xA = B, or A.T x.T = B.T
	A = Q + 10**-5*np.eye(num_features)
	B = P#[:num_features - 1,:] #TODO maybe shouldn't subtract 1 (since then wouldn't be corrupting last real feature, instead of not corrupting bias)
	mapping = np.linalg.solve(A.transpose(), B.transpose())#.transpose()
	return mapping

#Stack mDA layers on top of each other, using previous layer as input for the next
#Can optionally pass in precomputed mapping to use to transform data
	#(e.g. if transforming test data with mapping learned from training data)
def mSDA(data, prob_corruption, num_layers, use_nonlinearity = True, precomp_mappings = None):
	num_data, num_features = data.shape
	mappings = list()
	representations = list()
	representations.append(data)
	#construct remaining layers recursively based on the output of previous layer
	if precomp_mappings is None:
		for layer in range(0, num_layers):
			mapping, representation = mDA(representations[-1], prob_corruption, use_nonlinearity)
			mappings.append(mapping)
			representations.append(representation)
	else:
		for layer in range(0, num_layers):
			mapping, representation = mDA(representations[-1], prob_corruption, use_nonlinearity, precomp_mappings[layer])
			representations.append(representation)
		mappings = precomp_mappings
	return mappings, representations

#test implementation
if __name__ == "__main__":
	train_data = np.load("train_data.npy")
	test_data = np.load("test_data.npy")
	prob_corruption = 0.3
	bias = False
	print "Shape of data: ", train_data.shape

	if bias: #we want to include a bias term
		#data = np.vstack( (data,np.ones((1,data.shape[1])) ) ) #add extra column of 1's to incorporate bias
		train_data = np.hstack( (train_data,np.ones((train_data.shape[0],1)) ) ) #add extra column of 1's to incorporate bias
		test_data = np.hstack( (test_data,np.ones((test_data.shape[0],1)) ) ) #add extra column of 1's to incorporate bias
	print "Shape of data after bias: ", train_data.shape

	train_mapping = compute_reconstruction_mapping(train_data, prob_corruption)
	print "Shape of mapping: ", train_mapping.shape

	train_map, train_rep = mDA(train_data, prob_corruption, train_mapping)
	print "Shape of mDA representation: ", train_rep.shape

	num_layers = 3
	train_mappings, train_reps = mSDA(train_data, prob_corruption, num_layers)
	print "Shape of mSDA train representation: ", train_reps[-1].shape

	test_mappings, test_reps = mSDA(test_data, prob_corruption, num_layers, train_mappings)
	print "Shape of mSDA test representation: ", test_reps[-1].shape

	from sklearn import svm
	train_labels = np.load("train_labels.npy")
	test_labels = np.load("test_labels.npy")

	clf_ordinary = svm.SVC().fit(train_data, train_labels)
	preds_ordinary = clf_ordinary.predict(test_data)
	ordinary_accuracy = np.mean(preds_ordinary == test_labels)
	print "Accuracy on regular data: ", ordinary_accuracy

	clf_deepRep = svm.SVC().fit(train_reps[-1], train_labels)
	preds_with_deepRep = clf_deepRep.predict(test_reps[-1])
	deep_accuracy = np.mean(preds_with_deepRep == test_labels)
	print "Accuracy with linear SVM on mSDA features: ", deep_accuracy

	clf_mcf = svm.SVC().fit(np.dot(train_data,train_map.transpose()), train_labels)
	preds_mcf = clf_mcf.predict(np.dot(test_data, train_map.transpose()) )
	mcf_accuracy = np.mean(preds_mcf == test_labels)
	print "Accuracy with linear SVM on mcf: ", mcf_accuracy

	'''
	(num_features, num_data, prob_corruption, num_layers) = (5000, 10000, 0.4, 3)
	print("%d-layer mSDA on %d examples with %d features" % (num_layers, num_data, num_features))
	before = time.time()
	deep_rep = mSDA(np.random.rand(num_features, num_data), prob_corruption, num_layers) 
	after = time.time()
	print("Performed mSDA in %f seconds" % (after - before))
	print deep_rep[0].shape
	'''
