# This RNN takes in preprocessed data in the form on numpy arrays
# and runs it through several LSTM layers with dropout in order
# to train a model that can predict sign language signs
#
# - Grant Fisher and Daniel Book

import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, LSTM, Dropout
from keras.layers import TimeDistributed, BatchNormalization
from keras.optimizers import Adam
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, precision_score
from sklearn.metrics import recall_score, accuracy_score
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

# import data
X_train = np.load('xtrain.npy')
Y_train = np.load('ytrain.npy')
X_test = np.load('xtest.npy')
Y_test = np.load('ytest.npy')

# build model
print("Building model")

model = Sequential()
model.add(LSTM(128, input_shape=(150, 390), return_sequences = True))
model.add(Dropout(0.5))                             
model.add(BatchNormalization())

model.add(LSTM(128, input_shape=(150, 390), return_sequences = True))
model.add(Dropout(0.5))                             
model.add(BatchNormalization())

model.add(LSTM(64, input_shape=(150, 390), return_sequences = True))
model.add(Dropout(0.5))                             
model.add(BatchNormalization())

model.add(LSTM(32, input_shape=(150, 390), return_sequences = True))
model.add(Dropout(0.5))                             
model.add(BatchNormalization())

model.add(TimeDistributed(Dense(11))) 
model.add(Activation('softmax'))
myOptimizer = Adam(lr = 0.001) 
model.compile(loss='categorical_crossentropy', optimizer=myOptimizer, metrics=['categorical_accuracy'])

# summarize model
print(model.summary())

# train model
model.fit(X_train, Y_train, batch_size=4, epochs = 50) 

# evaluate
loss, acc = model.evaluate(X_test, Y_test)

print("Test set accuracy = ", acc)
print("Test set loss = ", loss)

# predict
predictions = model.predict(X_test)


# evaluate the model using custom metrics.  Because we have a multiclass problem that
# outputs probabilities, we have to loop through the output matrix to compute common
# metrics like recall score, f1 score, and a confusion matrix.  For the confusion
# matrix, we convert the one hot encoding back into integer form for ease of 
# viewing in a figure.

# initialize metrics
f1 = 0 # f1 score
f1WithZeros = 0 # f1 score if we just always predicted 'no sign'
rs = 0 # recall score
ra = 0 # ROC AUC score
acs = 0 # accuracy score
cm = np.zeros((11,11)) # confusion matrix
allZeroPredictions = np.zeros((11,1))
allZeroPredictions[0] = 1
numExamples = 0

for i in range(predictions.shape[0]):
	for j in range(predictions.shape[1]):
		index = np.argmax(predictions[i,j,:])
		predictionsOneHot = np.zeros((11,1))
		predictionsOneHot[index] = 1
		f1 += f1_score(Y_test[i,j,:], predictionsOneHot, average='macro')  
		f1WithZeros += f1_score(Y_test[i,j,:], allZeroPredictions, average='macro') 
		rs += recall_score(Y_test[i,j,:], predictionsOneHot)
		ra += roc_auc_score(Y_test[i,j,:], predictionsOneHot)
		acs += accuracy_score(Y_test[i,j,:], predictionsOneHot)

		Y_TestInteger = np.zeros((1,1))
		predictionsInteger = np.zeros((1,1))
		Y_TestInteger[0] = np.argmax(Y_test[i,j,:])

		predictionsInteger[0] = np.argmax(predictionsOneHot)


		cm = np.add(cm, confusion_matrix(Y_TestInteger, predictionsInteger, labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
		numExamples = numExamples + 1

# print scores so we can compare the performance of models
print("F1 score of actual predictions = ", (f1/numExamples))
print("F1 score of predicting all 0s = ", (f1WithZeros/numExamples))
print("Recall score: ", rs/numExamples)
print("ROC AUC score: ", ra/numExamples)
print("Accuracy score: ", acs/numExamples)
print("Confusion matrix: ", cm)

# normalize for ease of viewing
cm = normalize(cm, axis=0, norm='l1')
cm = normalize(cm, axis=1, norm='l1')

# plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()






