import pandas as pd
import keras
import sklearn
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

feats = ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
features = ['Survived', 'Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Embarked']
keyFeatures = ['Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Embarked']

data = pd.read_csv("train.csv", na_values=['?'], names=feats)
data = data[features]

#data = pd.read_csv("train.csv", usecols=[1,2,4,5,6,7,11])
data.dropna(inplace=True)  #   Drop entries with missing values

data.replace(('S','C','Q'), (1,2,3), inplace=True)
data.replace(('male','female'),(1,2), inplace=True)
#   We have trainTargets which are all 1 or 0 values.
#   We have trainData, which comprises of the features I've decided are key to selection for my algorithm.

#print(data.head())
#print(data.describe())
#print(data)    --  Used to interrogate pulled data - 715 entries over 7 columns - 1 column for target, 6 for features

trainFeatures = data[keyFeatures].values    #   Pulls the 6 features into training features
trainTargets = data['Survived'].values      #   Pulls our target labels into one place

trainFeatures = trainFeatures[1:]
trainTargets = trainTargets[1:]

#   Create the learning model now
def create_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(48, activation='tanh', kernel_initializer='normal', input_dim=6))  #   Input of 6 features
   # model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(24, activation='tanh', kernel_initializer='normal'))
   # model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1, activation='sigmoid', kernel_initializer='normal'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=create_model, nb_epoch = 100, verbose=0)

cvs = cross_val_score(estimator, trainFeatures, trainTargets, cv=10)

print(cvs.mean())