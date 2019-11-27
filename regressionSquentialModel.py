import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from category_encoders import *
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, MinMaxScaler,
                                   OneHotEncoder, OrdinalEncoder,
                                   StandardScaler)
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import metrics

def changeSizeOfCity(dataset):
    dataset['Small City'] = dataset['Size of City']
    dataset['Small City'] = dataset['Size of City']
    dataset.pop('Size of City')
    return dataset


def work_experience(dataset):
    dataset['Work Experience in Current Job [years]'] = dataset['Work Experience in Current Job [years]'].replace('#NUM!', 0, inplace= True)
    dataset['Work Experience in Current Job [years]'] = np.log(dataset['Work Experience in Current Job [years]'].astype('float'))

    return dataset


def processAdditionToSalary(dataset):
    dataset['Yearly Income in addition to Salary (e.g. Rental Income)'] = dataset['Yearly Income in addition to Salary (e.g. Rental Income)'].map(lambda x: x.rstrip(' EUR'))
    dataset['Yearly Income in addition to Salary (e.g. Rental Income)'] = np.log(dataset['Yearly Income in addition to Salary (e.g. Rental Income)'].astype('float'))

    return dataset


def degree(dataset):
    dataset["University Degree"] = dataset["University Degree"].replace(np.nan, "MISSING")
    dataset["University Degree"] = dataset["University Degree"].replace(0, "MISSING")
    dataset["University Degree"] = dataset["University Degree"].replace("0", "MISSING")
    return dataset


def genderCleaning(dataset):
    dataset["Gender"] = dataset["Gender"].replace("f", "female")
    dataset["Gender"] = dataset["Gender"].replace("0", "unknown")
    dataset["Gender"] = dataset["Gender"].replace(0, "unknown")
    dataset["Gender"] = dataset["Gender"].replace(np.NaN, "unknown")
    return dataset


def satisfaction(dataset):
    dataset['Satisfation with employer'] = dataset['Satisfation with employer'].replace(np.nan, "MISSING")
    return dataset


def age(dataset):
    dataset['Age'] = np.log(dataset['Age'])
    return dataset


def year(dataset,isTestData):
    if(isTestData==False):
        dataset["Year of Record"] = dataset["Year of Record"].replace(np.nan, 1990)
        dataset["Year of Record"] = dataset["Year of Record"].replace('#N/A', 1990)
        dataset["Year of Record"] = dataset["Year of Record"]
    else:
        dataset["Year of Record"] = dataset["Year of Record"].replace(np.nan, 1990)
    return dataset


def bodyHeight(dataset):
    return dataset


def profession(dataset):
    dataset['Profession'].fillna('none', inplace=True)
    return dataset


def housing(dataset):
    dataset["Housing Situation"] = dataset["Housing Situation"].replace(0, "Unknown")
    dataset["Housing Situation"] = dataset["Housing Situation"].replace("0", "Unknown")
    dataset["Housing Situation"] = dataset["Housing Situation"].replace("nA", "Unknown")
    return dataset

def one_hot_encoder(data,column):
    one_hot = pd.get_dummies(data[column])
    data = data.drop(column,axis = 'columns')
    return pd.concat([data,one_hot],axis='columns')


def build_model(training_dataset):
    # Sequential model used with two dense layers of 64 nodes
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(training_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),

    layers.Dense(1)  # output layer
  ])
  #optimizer = tf.keras.optimizers.RMSprop(0.001)
  model.compile(loss='mse',
                optimizer='adam',
                metrics=['mae', 'mse'])
  model.summary()

  return model

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(hist['epoch'], hist['mean_absolute_error'], label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label = 'Validation Error')
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error')
  plt.plot(hist['epoch'], hist['mean_squared_error'], label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'], label = 'Validation Error')
  plt.legend()
  plt.show()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('loss')
  plt.plot(hist['epoch'], hist['loss'], label='Train Error')
  plt.plot(hist['epoch'], hist['val_loss'], label = 'Validation Error')
  plt.legend()
  plt.show()

def printResults(test_targets, test_predictions):
    print('Mean Absolute Error:', metrics.mean_absolute_error(test_targets, test_predictions))
    print('Mean Squared Error:', metrics.mean_squared_error(test_targets, test_predictions))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_targets, test_predictions)))


#############################################################################
dataset = pd.read_csv('tcd-ml-1920-group-income-train.csv')
test_dataset = pd.read_csv('tcd-ml-1920-group-income-test.csv')

predict_X = test_dataset
train = dataset

train = train[:1044560]
train = train.drop_duplicates(subset='Instance', keep='first',
                              inplace=False)


y = train['Total Yearly Income [EUR]']

train = train.drop(columns=['Instance',
                            'Hair Color',
                            'Crime Level in the City of Employement',
                            'Total Yearly Income [EUR]',
                            ])
train = bodyHeight(train)
train = changeSizeOfCity(train)
train = degree(train)
train = year(train,False)
train = genderCleaning(train)
train = profession(train)
train = work_experience(train)
train = processAdditionToSalary(train)
train = housing(train)

cols_to_norm = ['Year of Record','Age','Body Height [cm]','Small City','Yearly Income in addition to Salary (e.g. Rental Income)']
train[cols_to_norm] = train[cols_to_norm].apply(lambda x: (x - x.mean()) / ( x.std()))

train = train.drop('Work Experience in Current Job [years]',axis='columns')
train = train.drop('Yearly Income in addition to Salary (e.g. Rental Income)',axis='columns')

#train.to_csv("out.csv")



#############################TESTDATA##########################################


predict_X = predict_X.drop(columns=['Instance',
                            'Hair Color',
                            'Crime Level in the City of Employement',
                            'Total Yearly Income [EUR]'
                            ])

predict_X = bodyHeight(predict_X)
predict_X = changeSizeOfCity(predict_X)
predict_X = degree(predict_X)
predict_X = genderCleaning(predict_X)
predict_X = year(predict_X,True)
predict_X = profession(predict_X)
predict_X = work_experience(predict_X)
predict_X = processAdditionToSalary(predict_X)

cols_to_normB = ['Year of Record','Age','Body Height [cm]','Small City','Yearly Income in addition to Salary (e.g. Rental Income)']
predict_X[cols_to_normB] = predict_X[cols_to_normB].apply(lambda x: (x - x.mean()) / ( x.std()))

predict_X = predict_X.drop('Work Experience in Current Job [years]',axis='columns')
predict_X = predict_X.drop('Yearly Income in addition to Salary (e.g. Rental Income)',axis='columns')
    # train, predict_X = train.align(predict_X , join='outer', axis=1, fill_value=0)

#######################ONEHOTENCODE############################################
train['train']=1
predict_X['train']=0

#Combine the dataset for one hot encoding
combined=pd.concat([train,predict_X],axis=0,sort=False)

combined=one_hot_encoder(combined,'Gender')
combined=one_hot_encoder(combined,'Profession')
combined=one_hot_encoder(combined,'University Degree')
combined=one_hot_encoder(combined,'Country')
combined=one_hot_encoder(combined,'Housing Situation')
combined=one_hot_encoder(combined,'Satisfation with employer')

train=combined.loc[combined['train'] == 1]
predict_X=combined.loc[combined['train'] == 0]
train=train.drop("train", axis='columns')
predict_X=predict_X.drop("train", axis='columns')


print(train.head(20))

# predict_X = predict_X.drop('Total Yearly Income [EUR]',axis='columns')
######################BUILDMODEL#################################################
X_train, X_test, y_train, y_test = train_test_split(
    train, y, test_size=0.2)

# Display progress prints single dot each go over the data
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.')

# number of itterations over the dataset
EPOCHS = 60

# Train the model
model = build_model(X_train)

# The patience parameter will check for loss and stop if loss becomes stagnant
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

# fit the model
history = model.fit(X_train, y_train, epochs=EPOCHS,
                    validation_split = 0.2, verbose=1,batch_size=70, callbacks=[early_stop, PrintDot()])

# print the history of the model
plot_history(history)

loss, mae, mse = model.evaluate(X_test, y_test, verbose=0)

print("Testing set Mean Abs Error:"+str(mae))

test_predictions = model.predict(X_test).flatten()

# create a scatter plot of the data
plt.scatter(y_test, test_predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
plt.plot([-100, 100], [-100, 100])

error = test_predictions - y_test
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error")
plt.ylabel("Count")
plt.show()

# show the results of predicted vs actual labels
printResults(y_test, test_predictions)

# save the evaluation results and submit to kaggle
# test_prediction = model.predict(normed_test_data_kaggle)
# print(test_prediction)
# np.savetxt("output-model-predictions.csv", test_prediction, delimiter=",")
submission_predictions = model.predict(predict_X).flatten()
np.savetxt("output-model-predictions.csv", submission_predictions, delimiter=",")
