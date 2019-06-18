#install keras. Go to anaconda propmp, run as admin and typ->  conda install -c conda-forge keras
#check help by selecting keywork and pressing CNTRL+I

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')        #In the dataset, only last coluumn is a dependent variable, all the rest are independent variable which affect the dependant variable
X = dataset.iloc[:, 3:13].values                 #X is the independant variable/or grop of variables. Here, the columns index,rownumber,customerId and surname have absoultely no effect on the dependant variable(Whethet the customer leaves the bank or not)
                                                    #The rest of the variables/columns however do have an impact so have to be considered. So we consider column indexes 3 to 13 and all the rows.If we do 3:13, it takes all columns 3 till 13, excluding 13
y = dataset.iloc[:, 13].values                      #Y is the dependent variable which is dependant on X(group of independent vaiables). Y is only the last column, whether the customer leaves the bank or not.


# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
                                                    #Here in our X we have only 2 categorical variables. Country which has 3 categories, france germany and spain , and gender which has 2 categories, male and female. So we will need 2 labelEncoder objects
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])   #Here, the object array X has the countries as 2nd column, but python arrays start from 0
#The above 3 lines will convert the strings into 3 numbers, each signifying a different string.
# We need to repeat the above step for "Male" and "Female" also
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])


onehotencoder = OneHotEncoder(categorical_features = [1])   #This helps us create the dummy variables.  Problem with label encoding is that it assumes higher the categorical value, better the category. This is why we use one hot encoder to perform “binarization” of the category and include it as a feature to train the model.Another Example: Suppose you have ‘flower’ feature which can take values ‘daffodil’, ‘lily’, and ‘rose’. One hot encoding converts ‘flower’ feature to three seperate features(3 seperate columns), ‘is_daffodil’, ‘is_lily’, and ‘is_rose’ which all are binary. 
X = onehotencoder.fit_transform(X).toarray()      #https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f # So now the whole X array became type float64 from object, as all columns are now of same type
X = X[:, 1:]
# So now to avoid falling in dummy variable trap, we remove one column of the 3 dummy variables we just created.

# Encoding the Dependent Variable
#labelencoder_y = LabelEncoder()
#y = labelencoder_y.fit_transform(y)       #We dont need to encode the Y dependent variable as it contains numerical values, 0 or 1

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)    


# Feature Scaling                                 We do feature scaling to help normalize all the values of the test and train set
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Now we make the ANN
import keras
from keras.models import Sequential    # Used to inintialize the neural network
from keras.layers import Dense     #We use this to create layers in ANN
from keras.layers import Dropout # We import this for dropout regularization to prevent overfitting if needed.

#Initializing the ANN(Defining it as sequence of layers) There are two ways of initializing, defining as layers or defining as graph
classifier=Sequential()         #Classifier is the name of the ANN

#In out input layer, we will have 11 input nodes as we have 11 independant variables
#See video number 25 in udemy Deep learning A-Z course

# Adding the input layer and the first hidden layer
#units is basically output_dim whihc is nodes in output of that particular layer.
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
#Here the input nodes is 11, and hidden layer nodes is 6.Normally the number of nodes in hidden layer is got by trial and error to optimize the network. But normally a good starting point is average of nodes in input layer and nodes in output layer. So here (11+1)/2
#Here the activation function we use is rectifier for hodden layers and sigmoid for output layers.

#Include below line only when using dropout to combat overfitting
classifier.add(Dropout(p=0.1))#Here p is the fraction of neurons to drop on each iteration. Generally p should be taken up to 0.5. after that insufficient neurons will be present to learn, and that will reslt in underfitting.


# Adding the second hidden layer    #Not really useful in this case, but its good to know how to do it
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))           #Here we dont need to define input nodes parameters as it knows this from the previous layer

#Include below line only when using dropout to combat overfitting
classifier.add(Dropout(p=0.1))#Here p is the fraction of neurons to drop on each iteration. Generally p should be taken up to 0.5. after that insufficient neurons will be present to learn, and that will reslt in underfitting.

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))        #Here output layer should have only 1 node
#here if dependant variable had 3 categories instead of 2(leave or not), then we would need to change two things, units would be 3, activation function would be "softmax". Softmax is sigmoid function applied to dependant variable which has more than 2 categories

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#Optimizer is algo used to fidn optimal set of weights to make the ANN powerful. Here adam is a stochastic gradient algo. Here the loss function is the cost/loss function used to optimize the weights. 'binary_crossentropy' is a logarithmic loss function which has only 2 categories. Metrics is the criteria chosen to ealuate your model. The aim of the ANN will be to optimize/increaze accuracy

# Do classifier.summary() to see the summary of teh neural network    See https://stats.stackexchange.com/questions/261008/deep-learning-how-do-i-know-which-variables-are-important

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)      # Batch size is the number of rows which are taken in each batch to avoid getting the local minima. See Stochastic Gradient Descent Video in Udemy Deep Learning A-Z course


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)                             #Predict method returns probabilities, but for the confusion matrix we need True/False. We take this as, if probability if leaving is more than 0.5, then put y_pred as true,otherwise False

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Total accuracy can be got by adding the main diagonal elements(true positive and true negative) of the confusion matrix and divide by total number in test set(2000 here). For me it is (1506+214)/2000 for this confusion matrix [[1506   89] [ 191  214]]

"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""

new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))       #We enter the respective values in the first row of a 2d array. We do this by opening two brackets, and taking the values only inside the inner bracket. Also france is taken as that to remove the dummy variable catch
new_prediction = (new_prediction > 0.5)
print(new_prediction)


# Part 4 - Evaluating, Improving and Tuning the ANN

# Evaluating the ANN

# We use a k-fold cross validator. Previously we used 80% of dataset as training set and 20% as test set. However to increase accuracy, we use k fold. Here we divide the dataset into groups of 10(say). Out of these any 9 will be training set and 1 will be test set. This will result in 10 combinations like this
#k fold cross validator is part of scikit but we need to apply it in our keras classifier, so there is a keras wrapper which integrates it.
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


from keras.models import Sequential
from keras.layers import Dense
#Also the KerasClassifier function needs a function as an arguement. So we define the classifier in the function. ONly the making of classifier, not fitting. It then returns the classifier
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)#We redefine classifier as the one in the function has ocal variable and cannot be used outside the function.

#cross_val_score returns 10 accuracies if dataset is divided into 10. We save these accuracies in accuracies variable.

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)    #CV define sthe number the dataset is split into. the n_jobs defines the number of CPUs taken. The number of CPUs to use to do the computation. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. 
# Previousl the full dataset was taken at once, then again for next epoch.
#Here, as the dataset is plit into 10, these 10 datasets will be operated upon simultaneously, and then again for next epoch...
mean = accuracies.mean()
variance = accuracies.std()

print(accuracies)
print(mean)
print(variance)
#If the above code doesnt work with spyder , run it directly in command propmpt,(after commenting appropriate code)

# Dropout Regularization to reduce overfitting if needed. Overfitting is when the classifier does well on training set but badly on test set and we can onserve this when we have a large difference of accuracies for training and test set.
#Another way to detect is when you get a high variance.
#Dropout-> On each iteration of training, some neurons are randomly disabled to prevent them from being too dependent on each other.  So now we get independent correlations of data whihc prevents neurons from learning too much and prevents overfitting
#So we import a new class dropout


# Tuning the ANN

#Initially, we have 2 kind of parameters, ones which change(like weights)  and some remain constant/hyperparameters(number of epoch,batch size, number of neurons)
#So parameter training consists changign these values to get the optimal neural network. We use GridSearch

#So we can copy the function and imports of the k-fold cross validation part, and instead import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)   #This keras classifier will not include epochs and batch size are these are the arguements we have to tune.

parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_