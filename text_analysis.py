#simple test of mSDA
import msda
import process_data
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
import numpy as np
import time

#TODO: fix up preprocessing data code and upload it as a separate file to WMD as well
#TODO: possibly use training mappings to transform test mSDA instead of performing mSDA again on test data
#		see about tanh in low dimensional approximation
#TODO: run more extensive tests on performance of mSDA with and without low dimensional approximation

#fetch training documents from 20 newsgroups dataset in random order
categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
all_20news = fetch_20newsgroups(subset='all',categories=categories, shuffle=True, random_state=42)
all_raw_data = all_20news.data #all the data
all_data_stringsList = process_data.createWordLists(process_data.unicodeToString(all_raw_data))
all_data_words = process_data.preprocess_by_word(all_data_stringsList)
all_raw_labels = all_20news.target #all the labels
all_full_data = process_data.vectorize(all_data_words) #convert to bag of words
all_full_data = all_full_data.transpose() #so rows are data, columns are features (format we predominantly use)
all_labels = all_raw_labels #already vectorized  #vectorize(all_raw_labels.tolist()) #convert to numeric labels
num_mostCommon = 800
all_mostCommonFeatures_data = process_data.getMostCommonFeatures(all_full_data, num_mostCommon)
train_data, train_labels, test_data, test_labels = process_data.splitTrainTest(all_mostCommonFeatures_data, all_labels)
#train_data = train_data.transpose() #for Chip's implementation of msda #TODO: debug
#test_data = test_data.transpose()

print "Shape of training data: ", train_data.shape
print "Shape of test data: ", test_data.shape

#classify with linear SVM
#transpose because sklearn requires (#data x #features)
clf_baseline = svm.SVC().fit(train_data.transpose(), train_labels)
baseline_preds = clf_baseline.predict(test_data.transpose())
base_accuracy = np.mean(baseline_preds == test_labels)
print "Accuracy with linear SVM on basic representation: ", base_accuracy

before_msda = time.time()
#learn deep representation with msda...
prob_corruption = 0.4
num_layers = 3
subproblem_size = 400

#need to transpose data to be in the right format (#features x #data) for mSDA
#specifically, the deep representation is the output from the last layer
#train_deepRep = msda.mSDA_lowDimApprox(train_data, prob_corruption, num_layers, subproblem_size)[1][:,:,-1]
#test_deepRep = msda.mSDA_lowDimApprox(test_data, prob_corruption, num_layers, subproblem_size)[1][:,:,-1]
train_deepRep = msda.mSDA(train_data, prob_corruption, num_layers)[1][:,:,-1]
#TODO: maybe should use same weights as on training features to transform test data?
test_deepRep = msda.mSDA(test_data, prob_corruption, num_layers)[1][:,:,-1]
#test_deepRep = np.tanh(np.dot(train_mapping, test_data))

#sklearn requires (#data x #features) so transpose back
train_deepRep = train_deepRep.transpose()
test_deepRep = test_deepRep.transpose()

after_msda = time.time()
print("used msda in %s seconds" % (after_msda - before_msda))
print "Shape of msda train rep: ", train_deepRep.shape
print "Shape of msda test rep: ", test_deepRep.shape

#...and classify with linear SVM
clf_deepRep = svm.SVC().fit(train_deepRep, train_labels)
preds_with_deepRep = clf_deepRep.predict(test_deepRep)
deep_accuracy = np.mean(preds_with_deepRep == test_labels)
print "Accuracy with linear SVM on mSDA features: ", deep_accuracy
