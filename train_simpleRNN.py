# http://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/
# LSTM with Variable Length Input Sequences to One Character Output
from build_LSTM import *
import pandas as pd
import numpy 
from random import shuffle
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Merge
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from theano.tensor.shared_randomstreams import RandomStreams
import pandas as pd
import random

max_len = 3 

def process_input(fn, poi_to_int, int_to_poi):
	'''convert input into designed format'''
	# load data
	data = pd.read_csv(fn,sep=" :",header = None)
	p_dataX = [map(int, x.split(' ')) for x in data[0].values.tolist()]
	p_dataY = data[1].values.tolist()
	dataX = []
	dataY = []
	for i in range(0,len(p_dataX)):
		sequence_in = p_dataX[i]
		sequence_out = p_dataY[i]
		tmp = [poi_to_int[poi] for poi in sequence_in]
		dataX.append(tmp)
		dataY.append(poi_to_int[sequence_out]-1)

 		if 'train' in fn:
 			for _ in range(0,2):
 				shuffle(tmp)
 				dataX.append(tmp)
 				dataY.append(poi_to_int[sequence_out]-1)

	#print int_to_poi
	# convert list of lists to array and pad sequences if needed
	X = pad_sequences(dataX, maxlen=max_len, dtype='float32',truncating='pre')
	y = np_utils.to_categorical(dataY,nb_classes=len(poi_to_int))
	return X, y


def mapping(train, test):
	'''build poi:index & index:poi map'''
	data = pd.read_csv(train,sep=" :",header = None)
	p_dataX = [map(int, x.split(' ')) for x in data[0].values.tolist()]
	p_dataY = data[1].values.tolist()
	
	data = pd.read_csv(test,sep=" :",header = None)
	p_dataX1 = [map(int, x.split(' ')) for x in data[0].values.tolist()]
	p_dataY1 = data[1].values.tolist()
	
	ax = p_dataX + p_dataX1
	ay = p_dataY + p_dataY1
		
	temp =sum(ax,ay)
	pois =set(temp)
	poi_to_int = dict(zip(pois,range(1,len(pois)+1)))
	int_to_poi = dict(zip(range(1,len(pois)+1),pois))
	return poi_to_int, int_to_poi

def ex_embedding():
	'''load embeddings from output(XG_ok:sequential preference, XP_ok:user preference) of PRME'''
	df = pd.HDFStore('sample4_fpmc_train_model_50.h5')
	print df 
	XG_ok = numpy.array(df['XG_ok'])
	XP_ok = numpy.array(df['XP_ok'])
	obj2id = dict(zip(df['obj2id'].Name, df['obj2id'].Id))
	index_dict = obj2id
	word_vectors = {}	
	for key in obj2id:
		tid = obj2id[key]
		word_vectors[key] = XG_ok[tid,:]
	return index_dict, word_vectors	

def read_sequence(fn):
    all_v = set()
    all_data = []
    all_pois = []
    with open(fn) as f:
    	for line in f:
    		tmp = line.strip().split(' ')
    		all_data.append(tmp)
    		all_v = all_v | set(tmp)
     		all_pois = all_pois + tmp
    pois = set(all_pois)
    poi_to_int = dict(zip(pois,range(1,len(pois)+1)))
    int_to_poi = dict(zip(range(1,len(pois)+1),pois))

    print '#venues:',len(all_v)
    print 'all_data:',len(all_data)
    
    sequence_length = 3 
        
    x_train = {}
    y_train = {}
    x_test = {}
    y_test = {}
    neg_pois = {}

    i=0
    for data in all_data:
    	train = int(round(len(data) * 0.8))
    	if train < sequence_length: 
    		result=data[0:train]
    		neg_pois_t = all_v - set(result)
    		x_train[i] = result[:-1]
    		y_train[i] = result[-1]

    		x_test_t = []
    		y_test_t = []	
    		for index in range(train,len(data)):
    			x_test_t.append(data[0:train])
    			y_test_t.append(data[index])	
    		x_test[i] = x_test_t 
    		y_test[i] = y_test_t 
    		neg_pois[i] = neg_pois_t	
    	else:
    		result = []
    		for index in range(len(data) - sequence_length):
    			result.append(data[index: index + sequence_length + 1])
    			if (index+sequence_length==train):
    				row = index
    			
    		result = np.array(result)
    		neg_pois[i] = all_v - set(data[0:train]) 	
    		result_tru = result[:row, :]
    		np.random.shuffle(result_tru)
    		x_train[i] = result_tru[:, :-1]
    		y_train[i] = result_tru[:, -1]
    		x_test_t = result[row:, :-1]
    		num_row,num_len = x_test_t.shape
    		x_test[i] = np.array([x_test_t[0,:],]*num_row) 
    		y_test[i] = result[row:, -1]
    	i = i+1

    return x_train, y_train, x_test, y_test, neg_pois, poi_to_int, int_to_poi

inputf = 'sample_users_all'

x_train, y_train, x_test, y_test, neg_pois, poi_to_int, int_to_poi= read_sequence(inputf)
print 'XXXXXXXXXX',len(x_train), len(y_train),len(x_test),len(y_test)
num_poi = len(int_to_poi) + 1
num_len = max_len
num_train = len(x_train)  
model, ppoi_embedding_model = build(num_train, num_len, num_poi)
print 'model input:',model.input_shape, model.output_shape


def batch_generator_train(x_train, y_train, x_test, y_test, neg_pois, poi_to_int, int_to_poi, batch_size):
    while True:
        uindex_batch = numpy.random.randint(len(x_train), size=batch_size)
        x_batch = [] 
        y_batch = [] 
        z_batch = [] 
        y_label = [] 

        for index in uindex_batch:
	    while len(x_train[index])==0:  
	        index =  random.randint(0,num_train-1)
	    sample_ind = random.randint(0,len(x_train[index])-1) 
            sequence_in = x_train[index][sample_ind]
            sequence_out = y_train[index][sample_ind]
            sequence_out2 = random.sample(neg_pois[index], 1)

            tmp = [poi_to_int[poi] for poi in sequence_in]
            x_batch.append(tmp)
            y_batch.append(poi_to_int[sequence_out])
            tmp = [poi_to_int[poi] for poi in sequence_out2]
            z_batch.append(tmp)
            y_label.append(1) 

        x_batch = numpy.asarray(x_batch)
        y_batch = numpy.asarray(y_batch) 
        z_batch = numpy.asarray(z_batch) 
        y_label = numpy.asarray(y_label) 
        x_label = [x_batch, uindex_batch, y_batch, z_batch]	
        yield x_label, y_label 

batch_size = 300 
n_epoch = 20000 
callbacks = [
]
generator=batch_generator_train(x_train, y_train, x_test, y_test, neg_pois, poi_to_int, int_to_poi, batch_size)
history = model.fit_generator(generator,samples_per_epoch=batch_size,nb_epoch=n_epoch)

che = model.layers[-1]
weights = che.get_weights()
U, V = weights[0], weights[1]

total_case = 0
correct_case = 0
topN = 5 
total_p = 0
total_r = 0
candidates_pois = range(0,len(int_to_poi)) 
candidates_pois = numpy.asarray(candidates_pois) 
ppoi_embedding = ppoi_embedding_model.predict([candidates_pois],batch_size=1000)

for ex in range(0,len(x_test)):
    user_id = ex

    ground_truth = [poi_to_int[poi] for poi in y_test[ex]]
    num_test = len(set(ground_truth))
    total_case = total_case + len(ground_truth) 
    predicted = [ (candidates_pois[i], np.sum(U[user_id,:]*ppoi_embedding[i,:]) ) for i in xrange(len(candidates_pois))]
    predicted = sorted(predicted, key=lambda i: i[1])[::-1]
    predicted = [i[0] for i in predicted]
    precision = len(list(set(predicted[:topN]) & set(ground_truth)))*1.0/topN
    recall = len(list(set(predicted[:topN]) & set(ground_truth)))*1.0/len(ground_truth)
    correct_case = correct_case + len(list(set(predicted[:topN]) & set(ground_truth)))
    total_p+=topN
    total_r+=len(set(ground_truth))
print correct_case, correct_case*1.0/total_case
print 'precision:',correct_case*1.0/total_p
print 'recall:',correct_case*1.0/total_r


