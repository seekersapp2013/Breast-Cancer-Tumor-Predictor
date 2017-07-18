# Import libraries
from keras.models import *
from keras.layers import *
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import numpy as np
import pandas as pd

###################### Hyperparameters ######################
# Number of different classes for classification
classes = 2

# Size of the data that is used every batch
batch_size = 2

# Number of the inputs that will be fed into the neural network
input_shape = (9,)

###################### Data Extraction ######################
df = pd.read_csv('cancer_data.csv')

# Drop any columns that contain "?" or no data
df = df.replace({'?': np.nan}).dropna()

# Split the data into testing and training data (80/20)
test, train = train_test_split(df, test_size = 0.8)

# Assign training and testing data
x_train, y_train = train.ix[:,0:9].as_matrix(), train.ix[:,9].as_matrix()
x_test, y_test = test.ix[:,0:9].as_matrix(), test.ix[:,9].as_matrix()

# Convert labels (Y values) to one-hot vectors
y_train = to_categorical(y_train, classes)
y_test = to_categorical(y_test, classes)

###################### Model Creation ######################
model = Sequential()
model.add(Dense(128, activation = 'relu', input_shape = input_shape))
model.add(Dropout(.2))
model.add(Dense(classes, activation = 'softmax'))
model.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy'])

###################### Model Testing ######################
model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=100,
              verbose=1,
              validation_data=(x_test, y_test))

###################### Model Prediction ######################
inputs = np.array([[3,10,7,8,5,8,7,4,1]])
prediction = model.predict(inputs)
print "Benign Probability: " + str(prediction[0][0])
print "Malignant Probability: " + str(prediction[0][1])
