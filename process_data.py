#Preprocess data (turn raw data into bag of words, split into training and test)
import numpy as np, sys, random, string, unicodedata

#Convert unicode corpus to strings
def unicodeToString(data):
	string_data = [] #remake dataset as a list of strings instead of a list of unicode
	for document in data:
		str_doc = str(unicodedata.normalize("NFKD",document).encode("ascii","ignore"))
		string_data.append(str_doc)
	return string_data

#Create word lists out of documents (a list of data): turn each document into a list of words
#Assume documents are originally in unicode; also convert to strings
def createWordLists(data):
	documents = []
	for document in data:
		doc_wordList = document.split() #split on whitespace (getting list of words)
		documents.append(doc_wordList)
	return documents

#Preprocess data one word at a time
def preprocess_by_word(data):
	stops_file = open("stop_words.txt", "r") #each (unique) stop word is on its own line, with extra white space for padding
	stop_words = set([line.strip() for line in stops_file if line != "\n"]) #strip off carriage returns and store stop words in a list
	preprocessed_docs = [] #remake the dataset in its preprocessed form (TODO modify in place?)
	for document in data:
		preprocessed_doc = []
		for word in document:
			if word not in stop_words: #only count words not in stop words
				#remove punctuation
				nopunct = word.translate(string.maketrans("",""), string.punctuation)
				preprocessed_doc.append(nopunct.lower()) #also lowercase
		preprocessed_docs.append(preprocessed_doc)
	return preprocessed_docs

#Read in data in file name for preprocessing and use: assumes data format is label separated by tab from words
#Also lowercase all words in case they aren't already (which in the twitter data, for example, they already are)
def preprocess(file_name):
	stops_file = open("stop_words.txt", "r") #each (unique) stop word is on its own line, with extra white space for padding
	stop_words = set([line.strip() for line in stops_file if line != "\n"]) #strip off carriage returns and store stop words in a list
	datafile = open(file_name, "r") #open the file containing the data for reading
	text_labels = [] #store labels as they are read in
	text_data = [] #store data as it is read in

	for datum in datafile: #each line represents a piece of data
		split = datum.split("\t")
		label = split[0].replace("\"", "") #label is separated by a tab (also get rid of quotation marks)
		text = split[1].lower().split() #words in document are on the other side. lowercase and store in list
		preproc_text = [word for word in datum if word not in stop_words] #remove stop words and leave what's left
		text_labels.append(label)
		text_data.append(text)

	#vectorize data and labels so that they can be fed into a machine learning algorithm
	return vectorize(text_data), vectorize(text_labels)

#Convert a list of (e.g.) textual features/labels into numbers so that it can be fed into a machine learning algorithm
def vectorize(data):
	if type(data[0]) is not list: #str: #data is a list (e.g. of labels) and not a list of lists (e.g. of features)
		return vectorize_list(data)

	vocabulary = dict()
	vocab_size = 0
	vectorized_docs = list() #list of all the vectorized documents
	for doc_wordList in data:
		document = [0] * vocab_size
		for word in doc_wordList:
			if word not in vocabulary: #we've seen a new word
				vocab_size += 1
				vocabulary[word] = vocab_size
				document.append(0) #keep expanding document size to match size of vocabulary seen so far 
			feature_number = vocabulary[word] #we know this is in the dictionary because if not we just added it
			document[feature_number - 1] += 1
		vectorized_docs.append(document)
	
	#Make sure all documents have the same size as the vocabulary (otherwise add 0's on to the end)
	for doc in vectorized_docs:
		while len(doc) < vocab_size:
			doc.append(0)
	
	#Combine vectorized documents into a Numpy array
	vec_data = np.array(vectorized_docs) #was np.asarray() but that just turned into an array of lists
	return vec_data

def vectorize_list(data):
	unique_entries = dict()
	num_unique = 0
	vectorized_features = list()
	for entry in data:
		if entry not in unique_entries: #we've seen a new word
			num_unique += 1
			unique_entries[entry] = num_unique
		feature_number = unique_entries[entry] #we know this is in the dictionary because if not we just added it
		vectorized_features.append(feature_number)
	return np.asarray(vectorized_features)

#split into training and test
#assumes features are rows, data points are columns
def splitTrainTest(data, labels):
	train_fraction = 0.8
	numData = data.shape[1]
	numTrain = int(numData * train_fraction)
	ordering = range(numData)
	random.shuffle(ordering)
	train_indices = ordering[:numTrain]
	test_indices = ordering[numTrain:]
	train_data = data[:,train_indices]
	train_labels = labels[train_indices]
	test_data = data[:,test_indices]
	test_labels = labels[test_indices]
	return train_data, train_labels, test_data, test_labels

'''Get specified number of most common features from a dataset'''
#assume features are rows and data points are columns
def getMostCommonFeatures(data, numFeatures):
	#if more features are requested than are in the original data just return original data
	if numFeatures >= data.shape[0]:
		return data
	#Sort rows by the total count in them
	sortedFeatureCounts = np.squeeze(np.asarray(data.sum(axis=1))).argsort() #make a vector of row sums, then sort it
	sortedByFeatureCount = data[sortedFeatureCounts]
	#Get the last "numFeatures" rows in this sorted array
	mostCommon = sortedByFeatureCount[-numFeatures:,:]
	return mostCommon