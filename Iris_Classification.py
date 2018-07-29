
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt



# load dataset-
dataset = pd.read_csv("iris.csv", header = None)
dataset = dataset.values
# remove first row as it's header-
dataset = dataset[1:,:]


# Split 'dataset' into 'train' & 'test' datasets-
train, test = train_test_split(dataset, test_size = 0.3)

# 'X' is input & 'y' is output-
train_X, train_y = train[:,0:4], train[:,4]
test_X, test_y = test[:,0:4], test[:,4]

train_X.shape	# O/P- (105, 4)
train_y.shape	# O/P- (105,)
test_X.shape	# O/P- (45, 4)
test_y.shape	# O/P- (45,)

# encode class values 'strings' as integers-
encoder = LabelEncoder()

encoder.fit(train_y)
encoder.fit(test_y)

encoded_train_y = encoder.transform(train_y)
encoded_test_y = encoder.transform(test_y)

encoded_train_y[:5]	# O/P- array([2, 2, 2, 1, 2])
encoded_test_y[:5]	# O/P- array([0, 0, 0, 0, 0])

# convert integer encoded values to dummy variables- one hot encoded:
dummy_train_y = np_utils.to_categorical(encoded_train_y)
dummy_test_y = np_utils.to_categorical(encoded_test_y)

dummy_train_y.shape	# O/P- (105, 3)
dummy_test_y.shape	# O/P- (45, 3)
train_X.shape		# O/P- (105, 4)
test_X.shape		# O/P- (45, 4)


# create model-
model = Sequential()
model.add(Dense(9, input_dim = 4, activation = "relu"))
# model.add(Dense(8, input_dim = 4, activation = "relu"))
# model.add(Dense(4, input_dim = 4, activation = "relu"))
model.add(Dense(3, activation = 'softmax'))

# compile model-
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

# Train/Fit the created model-
history = model.fit(train_X, dummy_train_y, epochs = 200, validation_data=(test_X, dummy_test_y))


# Plot graph for error during traing and validation-
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


# Make predictions-
preds = model.predict(test_X)

preds.shape	# O/P- (45, 3)


pred_values = []
 
for i in range(0, 45):
	l = 0 # largest values
	c = 0
	l = preds[i, 0]
	for j in range(1, 3):
		if preds[i, j] > l:
			l = preds[i, j]
			c = j
	# pred_values.append(l)
	pred_values.append(c)


# Predicted values for 'test_y'-
pred_test_y = np.array(pred_values)

"""
Encoding Scheme-
'Iris-setosa' -> 0
'Iris-versicolor' -> 1
'Iris-virginica' -> 2
"""



# Make confusion matrix to test model accuracy-
cm = confusion_matrix(encoded_test_y, pred_test_y)

accuracy = sum(cm.diagonal()) / cm.sum()

print("\n\nAccuracy achieved = {0:4f}\n\n".format(accuracy * 100))



