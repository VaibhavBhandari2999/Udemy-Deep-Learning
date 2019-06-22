#See https://www.udemy.com/deeplearning/learn/lecture/8374814#questions/3554002 formultiple indicators


# We will implement an LSTM to predict upward and downward trend of google stock prices.
# This LSTM will be a complex stacked LSTM with robust functionality. We will also use dropout
# The LSTM willl train on google stock price from 2012 to 2016 and will predict stock price of first month of 2017.

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt#To visualize
import pandas as pd#To import dataset

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')     #The RNN imports only the training set data
training_set = dataset_train.iloc[:, 1:2].values                #The iloc method is used to get the column indexes. We should consider all the rows and only the 1st column(open) array, which we do by defining 1:2. Now we add the '.values' to make it a numpy array. A NN can only take in numpy arrays.

# Feature Scaling

#Two best ways of feature scaling are standardizaton and nomrlaization. In standardization, we do (x-mean(x))/standard deaviation(x). Normalization is (x-min(x))/(max(x)-min(x)).
#In this RNN however, we use normalization mathod of featurs scaling. 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))       # the new feature range will be between 0 and 1, as all the new stock prices got by normalization are between 0 and 1.
training_set_scaled = sc.fit_transform(training_set)#Scaled training set is the nomalized training set 


# Creating a data structure with 60 timesteps and 1 output
# Now we create a datas tructure specifying what the RNN will need to predict next stock prices. This is the time stamps.

#This means the RNN will conisder the stock prices for 60 days before the current time stamp, and based on those values will try to predict the next output.
#60 timestamps is got by trial and error. 1 timestamp leads to overfitting(the model not learning anything), all the timstamps between 1 and 60 were not enough to capture the trends.
#So basically for each day we look at the 3 previous months(60timestamps /20 working days in a month) to preditc the output. 
#For each observational day, X-Train will contain 60 stock prices, while y_train will contain stick price of next financial day.
X_train = []
y_train = []
for i in range(60, 1258):               #As, for each value, we take 60 previous stock prices, we can only start doing it from the 60th stock price. We do it till 1258 as 1257 is the last index of the training_set.
    X_train.append(training_set_scaled[i-60:i, 0])  #We consider the 60 previous stock prices of the financial day. S rows are 60(from i-60) to i.The column is 0(which is the only columns present in training_set_scaled).
    y_train.append(training_set_scaled[i, 0])       #y_train simply needs the stock prices a ]t time t+1. It will be i, not i+1, as index starts from 0.
    
    #Till now X_train and y_train are lists which need to be converted to numpy arrays to be accepted by the RNN.
X_train, y_train = np.array(X_train), np.array(y_train)         # We can split this into 2 different lines

# Reshaping

#Till now we had only one indicator(independant variable) which we include to predict the stock price. By increasing the dimensions we can add more indicators(if we want).
X_train =  np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  #First arguement is numpy array to be reshaped. Second arguement is 
# See imput shape in https://keras.io/layers/recurrent/
#Right now the numpy array is a 2D array which is 1198*60. We add a third dimension, so it becomes a 3D array. Which is basically N numpy arrays of 1198*60 stacking on top of each other.
#The 2nd arguement of the np.reshape also contains 3 arguements. The first of these corresponds to the batch size(in our case, the number of observations). We have 1198 observations, but to make the code more general(so that it is applicable to any size), we take X_train.shape[0]. The second arguement is number of time stamps, which is number of values in the horizontal direction in the X_train dataset, whihc is 60 here, but to make the code more general, we take X_train.shape[1]. The third arguement shows the number of indicators/predictors which is 1(the open google stock price).
#As shape parameter gives the order of the dataset basically. IF it was a 3-D set, there will be a X_train.shape[2] also.

#A new X_train is formed, with 3 axes. We can see each axes by double clicking on the variable and changing the axes, at the nottom left(in spyder).

         
# Part 2 - Building the RNN

# Importing the Keras libraries and packages
#We're going to make  arobust stacked LSTM with dropout to prevent overfitting
from keras.models import Sequential#To allow us to create a sequentialobject with a seqence of layers
from keras.layers import Dense#To add output layers
from keras.layers import LSTM#To add LSTM layers
from keras.layers import Dropout#To add dropout reqularization

#Pytorch is much more powerful.
########----------------------------------- 
#Also now it is regressor, not classifier as we are preidtcing based on 1 trend, so we're doing some regression. 
#Regression is about predicting a continous value and classification is about preciting a category/class.
########-----------------------------------
# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))       #Units is the number of LSTM cells/units you want to have in this LSTM layer. For us, we need a large number of neurons as we need a model with high dimensioanlity as predicting stock price is pretty complex.If we have too less neurons, it wont be able to capture the upward and downward trend.Return Sequences should be true as when you have a stacked LSTM, one on top of another, retrun seqences should be true. When there is no LSTM layer after the one you're defining, you will set it to false(or no need to define it at all, as by default it is set to false). Input shape will be the 3 dimensions defined at line 44. But here we will only take the second and third arguement as the first one is automatically taken.
regressor.add(Dropout(0.2)) #Here p is the fraction of neurons to drop on each iteration. Generally p should be taken up to 0.5. after that insufficient neurons will be present to learn, and that will reslt in underfitting.

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True)) # We dont need to specify the input_shape as it understand this automatically from the previous layer.
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50)) #Here , the return_sequences will be False. Return Sequences should be true as when you have a stacked LSTM, one on top of another, retrun seqences should be true. When there is no LSTM layer after the one you're defining, you will set it to false(or no need to define it at all, as by default it is set to false)
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1)) #To make  afull connection between last LSTM layer and the output layer, we need to use the Dense class. The output layer needs to have just 1 neuron.

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
#See https://keras.io/optimizers/
# RMSprop optimizer is recommended for RNN(as seen in keras.io/optimizers documentation).However, We us optimizer as 'adam' is used  due to trial and error. 
#There is no activation function defined as the default value for activation function is used,ie linear.

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32) #Same as in ANN
#The loss will reduce in each iteration and in the last 20/30 epochs the loss wont reduce much


# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values      #Change it to numpy array
#Contains only 20 observations as only 20 financial days in a month are considered.

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)#This will contain whole training set and test set. It contains two arguements. Second one is whther we want to concatenate the rows or he columns.For horizontal concatenation, we use axis=1 and for vertical we use axis=0. The first arguement contains the two datasets we want to concatenate, but we want to join only the 'Open' parameter(for the Open_google_stock_price) so we do dataset_train['Open'] and dataset_test['Open'].
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values# So from dataset_total, we need the lower bound of the range of inputs we need. The lower bound is the stock price at Jan 3rd - 60. To get January 3rd,we do len(dataset_total) - len(dataset_test). As Dataset_total is train+test, this expression gives basically the index of the firts financial day of the new year, which is basically the first index of the test set(but as we now want to use dataset_total, we use thi sway to find the index of the first test set in dataset_total). Upper bound is the last index, whihc is denoted by a blank. Also, we use .values to make it a numpy array
inputs = inputs.reshape(-1,1) # As we didn't use illoc to get the numpy array, we have to reshape it.
#and you will often get this format problem,this shape problem when working with NumPy,and the solution to this,if you don't get this NumPy Array of the format you want,that is with your observations and linesand one or several columns,well, the trick to reshape the inputsis to use the reshape function,
# Also, we need to scale out inputs as we had scaled the dataset_train/X_train
#We only need to scale the inputs not the actual test values, as we need to keep the test values as they are.
inputs = sc.transform(inputs)# We dont use sc.fit, as we need to use the same way the sc affected the train to affect the total also., so we directly transorm, without fitting(Fitting basically fits the sc to that particular dataset in ordder to determine how to scale it better). 
X_test = []         #like X_train but for test
for i in range(60, 80):             #Lower bound should be 60 only, as i-60 shouldnt be invalid. As test set contains only 20 financial days, we only need to go from 60 to 80.
    X_test.append(inputs[i-60:i, 0])        #The 0 corresponds to the firts column of the inpu
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))#We get the 3d format here too.See line 46 onwards
predicted_stock_price = regressor.predict(X_test) 
predicted_stock_price = sc.inverse_transform(predicted_stock_price)# We need to inverse the scaling as the 
#It is of course to inverse the scaling of our predictions because of course our regressor was trained to predict the scaled values of the stock price, but no worries.To get the original scale of these scaled predicted values we simply need to apply the inverse transform method from our scaling sc object.

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


#Our model reacts well to smooth changes but not so well to fast non-linear changes.