#Implement mDA (Chen et. al, 2012)
#Take in data and probability of corruption
#"Corrupt" data (but marginalize out the (expected) corruption) and learn a reconstruction specified by weights
import numpy as np
from numpy import *
import numpy.matlib
import math, time


#Learn a deep representation of data by reconstructing "corrupted" input but marginalizing out corruption
#data format: features are rows, data points are columns
#Can optionally pass in precomputed mapping to use to transform data
	#(e.g. if transforming test data with mapping learned from training data)
def mDA(data, prob_corruption, mapping = None):
	#need to add on bias either way
	data = np.vstack( (data,np.ones((1,data.shape[1])) ) ) #add extra column of 1's to incorporate bias
	if mapping is None:
		num_features = data.shape[0] 

		#Represents the probability that a given feature will be corrupted (note that the bias term will never be corrupted)
		#feature_corruption_probs = np.append(np.ones((num_features - 1, 1))*(1-prob_corruption), 1)
		feature_corruption_probs = np.vstack( (np.ones((num_features - 1, 1))*(1-prob_corruption), 1) )
		scatter_matrix = np.dot(data, data.transpose())
		Q = scatter_matrix*(np.dot(feature_corruption_probs, feature_corruption_probs.transpose()))
		Q[np.diag_indices_from(Q)] = feature_corruption_probs[:,0] * np.diag(scatter_matrix)
		P = scatter_matrix * numpy.matlib.repmat(feature_corruption_probs, 1, num_features).transpose()

		#solve equation of the form x = BA^-1, or xA = B, or A.T x.% = B.T
		A = Q + 10**-5*np.eye(num_features)
		B = P[:num_features - 1,:]
		mapping = np.linalg.solve(A.transpose(), B.transpose()).transpose()
	representation = np.tanh(np.dot(mapping,data)) #inject nonlinearity
	return (mapping, representation)

#Stack mDA layers on top of each other, using previous layer as input for the next
#Can optionally pass in precomputed mapping to use to transform data
	#(e.g. if transforming test data with mapping learned from training data)
def mSDA(data, prob_corruption, num_layers, precomp_mappings = None):
	if precomp_mappings is None: #print this out the first time
		print "mSDA"
	(num_features, num_data) = np.shape(data)
	mappings = np.zeros((num_features, num_features + 1, num_layers))
	representations = np.zeros((num_features, num_data, num_layers + 1))
	representations[:,:,0] = data #first layer is the original data
	#construct remaining layers recursively based on the output of previous layer
	if precomp_mappings is None:
		for layer in range(0, num_layers):
			(mappings[:,:,layer], representations[:,:,layer+1]) = mDA(representations[:,:,layer],prob_corruption)
	else:
		for layer in range(0, num_layers):
			(mappings[:,:,layer], representations[:,:,layer+1]) = mDA(representations[:,:,layer],prob_corruption, precomp_mappings[:,:,layer])
	return (mappings, representations)

#Approximation of mSDA for high dimensional data
#breaking it up into smaller subproblems and averaging their solutions together
#take in data, mSDA hyperparameters, and how big at most you want your subproblems to be
def mSDA_lowDimApprox(data, prob_corruption, num_layers, max_subproblem_dimensionality, subprob_mappings = None, subseq_mappings = None):
	if subprob_mappings is None: #print this out the first time
		print "low dimensional approximation"
	dimensionality, num_data = np.shape(data) #in data matrix columns are data points
	if max_subproblem_dimensionality > dimensionality: #"subproblem" isn't a lower dimensional approximation
		return mSDA(data, prob_corruption, num_layers)

	#order in which we group off data points into subproblems
	random_feature_ordering = np.random.permutation(dimensionality)
	#try to make subproblems as even in size as possible:
	#figure out how many subproblems can be made of the given dimensionality
	num_subproblems = math.ceil(float(dimensionality) / max_subproblem_dimensionality)
	#given the desired number of subproblems, figure how many can be made out of the original data
	subproblem_dim = math.ceil(float(dimensionality)/num_subproblems) 

	#save data and learned mappings of the various subproblems
	subproblemData_list = []
	subproblemMappings_list = []

	#keep track of how many features have been considered in some subproblem
	#want to consider as many features as possible while keeping subproblems of equal size (so they can be avg-ed)
	num_featuresConsidered = 0 
	while num_featuresConsidered + subproblem_dim <= dimensionality: #make sure we don't overshoot
		#consider the next (subproblem-size) data points
		subproblem_featureIndices = \
				random_feature_ordering[num_featuresConsidered:num_featuresConsidered + subproblem_dim]
		subproblem_data = data[subproblem_featureIndices,:] #get the data of the points in this subproblem
		#learn a mapping for this subspace, and save the data and mapping (unless we already have mappings to use)
		if subprob_mappings is None: #no precomputed mappings to use
			subproblem_mapping = mDA(subproblem_data,prob_corruption)[0]
			subproblemMappings_list.append(subproblem_mapping)
		#to account for bias, just like in mDA, add "extra feature" of 1's
		subproblem_data = np.vstack( (subproblem_data,np.ones((1,subproblem_data.shape[1])) ) )
		subproblemData_list.append(subproblem_data)
		num_featuresConsidered += subproblem_dim #we have now finished considering more points

	#if we have precomputed subproblem mappings to use, use them
	if subprob_mappings is not None:
		subproblemMappings_list = subprob_mappings

	#used the learned mapping to reconstruct each subproblem and point wise add the reconstructions together
	sum_of_reconstructions = np.dot(subproblemMappings_list[0], subproblemData_list[0])
	for subprob in range(1,len(subproblemData_list)):
		sum_of_reconstructions += np.dot(subproblemMappings_list[subprob], subproblemData_list[subprob])

	#average and add nonlinear transformation
	avg_of_reconstructions = 1/float(num_subproblems) * sum_of_reconstructions
	firstLayer_output = np.tanh(avg_of_reconstructions) #inject nonlinearity as we would at the end of mDA

	#as with mSDA, construct remaining layers recursively based on the output of previous layer
	mappings = np.zeros((subproblem_dim, subproblem_dim + 1, num_layers))
	representations = np.zeros((subproblem_dim, num_data, num_layers + 1))
	representations[:,:,0] = firstLayer_output #first layer is now the output we calculated above
	#construct remaining layers recursively based on the output of previous layer
	#doesn't matter what the mapping is for the first layer (though we calculated it above)
	if subseq_mappings is None: #use mDA to learn remaining layers
		for layer in range(0, num_layers):
			(mappings[:,:,layer], representations[:,:,layer+1]) = mDA(representations[:,:,layer],prob_corruption)
	else: #use precomputed mappings for subsequent layers
		for layer in range(0, num_layers):
			(mappings[:,:,layer], representations[:,:,layer+1]) = mDA(representations[:,:,layer],prob_corruption, subseq_mappings[:,:,layer])
	return (subproblemMappings_list, mappings, representations)

#test implementation
if __name__ == "__main__":
	(num_features, num_data, prob_corruption, num_layers) = (5000, 350000, 0.4, 3)
	print("%d-layer mSDA on %d examples with %d features" % (num_layers, num_data, num_features))
	before = time.time()
	deep_rep = mSDA(np.random.rand(num_features, num_data), prob_corruption, num_layers) 
	after = time.time()
	print("Performed mSDA in %f seconds" % (after - before))
	print deep_rep[0].shape
