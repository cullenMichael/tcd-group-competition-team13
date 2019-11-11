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

# from xgboost import XGBRegressor


def imputation(train):
    train['Work Experience in Current Job [years]'].replace('#NUM!', np.nan, inplace= True)

    ord_si_step = ('si', SimpleImputer(strategy='constant',
                                       fill_value='MISSING'))
    ord_oe_step = ('oe', OrdinalEncoder())
    ord_steps = [ord_si_step, ord_oe_step]
    ord_pipe = Pipeline(ord_steps)
    ord_cols = ['Satisfation with employer']#,'University Degree' 'Housing Situation']
    #ord_cols = []

    cat_si_step = ('si', SimpleImputer(strategy='constant',
                                       fill_value='MISSING'))
    cat_ohe_step = ('te', MEstimateEncoder())
    cat_steps = [cat_si_step, cat_ohe_step]
    cat_pipe = Pipeline(cat_steps)
    cat_cols = ['Country', 'Profession'] # 'Hair Color', 'Gender'

    num_cols = ['Year of Record', 'Age', 'Body Height [cm]', 'Work Experience in Current Job [years]']
    num_si_step = ('si', SimpleImputer(strategy='median'))
    num_ss_step = ('ss', MinMaxScaler(feature_range=(0, 1)))#StandardScaler())
    num_steps = [num_si_step, num_ss_step]
    num_pipe = Pipeline(num_steps)

    transformers = [('cat', cat_pipe, cat_cols),
                    ('num', num_pipe, num_cols),
                    ('ord', ord_pipe, ord_cols)]
    return ColumnTransformer(transformers=transformers)


def Scaler(X_train):
    transformer = FunctionTransformer(np.log1p, validate=True)
    X_train = transformer.transform(X_train)

    scaler = MinMaxScaler(feature_range=(0, 1))

    scaler.fit(X_train)
    return scaler.transform(X_train)


def changeSizeOfCity(dataset):
    # Possibly refine number for this
    dataset['Small City'] = dataset['Size of City'] < 3000;
    dataset.pop('Size of City')
    return dataset


def processAdditionToSalary(dataset):
    # Get rid of "EUR" and change to float
    dataset['Yearly Income in addition to Salary (e.g. Rental Income)'] = dataset['Yearly Income in addition to Salary (e.g. Rental Income)'].map(lambda x: x.rstrip(' EUR'))
    dataset['Yearly Income in addition to Salary (e.g. Rental Income)'] = dataset['Yearly Income in addition to Salary (e.g. Rental Income)'].astype('float')
    dataset['Salary'] = dataset['Total Yearly Income [EUR]'] - dataset['Yearly Income in addition to Salary (e.g. Rental Income)']
    return dataset


# Might scrap this - no improvement in MAE
def degree(dataset):
    dataset['Has Degree'] = dataset['University Degree'].str.contains(pat = 'Bachelor|PhD|Master')
    dataset["University Degree"] = dataset["University Degree"].replace(np.nan, "MISSING")
    dataset["University Degree"] = dataset["University Degree"].replace(0, "MISSING")
    #dataset.pop('University Degree')
    return dataset


def HousingSituation(dataset, isTestData):
    # replace 0 and nA with common aribale "Unknown"
    dataset["Housing Situation"] = dataset["Housing Situation"].replace(0, "Unknown")
    dataset["Housing Situation"] = dataset["Housing Situation"].replace("0", "Unknown")
    dataset["Housing Situation"] = dataset["Housing Situation"].replace("nA", "Unknown")

    # if training data then drop columns otherwise leave them in
    if(isTestData is not True):
        for (i, row) in dataset.iterrows():
            if row["Total Yearly Income [EUR]"] > 400000:
                if row["Housing Situation"] in ("Castle", "Unknown", "Large House"):
                    dataset = dataset.drop(i, axis=0)

    # Shuffle dataset to ensure forward propagation
    dataset = shuffle(dataset)
    # Foward fill the the data
    dataset = dataset.replace({'Housing Situation': {"Unknown": np.nan}}).ffill()
    # resort the data to original ordering
    dataset = dataset.sort_values(by=['Instance'])

    return dataset


def genderCleaning(dataset, isTestData):
    # replace 0 and nA with common aribale "Unknown"
    dataset["Gender"] = dataset["Gender"].replace("f", "female")
    dataset["Gender"] = dataset["Gender"].replace("0", "unknown")
    dataset["Gender"] = dataset["Gender"].replace(0, "unknown")
    dataset["Gender"] = dataset["Gender"].replace(np.NaN, "unknown")

    # if training data then drop columns otherwise leave them in
    if(isTestData is not True):
        for (i, row) in dataset.iterrows():
            if row["Gender"] == "unknown":
                if row["Total Yearly Income [EUR]"] > 1500000:
                    dataset = dataset.drop(i, axis=0)

    # Shuffle dataset to ensure forward propagation
    dataset = shuffle(dataset)
    # Foward fill the the data
    dataset = dataset.replace({"Gender": {"unknown": np.nan}}).ffill()
    # resort the data to original ordering
    dataset = dataset.sort_values(by=["Instance"])

    return dataset


# No Missing values in Body Height so no need for Backward propagation
def bodyHeight(dataset):
    std = dataset['Body Height [cm]'].std(axis = 0)
    mean = dataset['Body Height [cm]'].mean()
    dataset['1StdBH'] = ((dataset['Body Height [cm]'] >= (mean - std)) & (dataset['Body Height [cm]'] <= (mean + std)))
    dataset['Outside 1 Std'] = ((dataset['Body Height [cm]'] <= (mean - std)) | (dataset['Body Height [cm]'] >= (mean + std)))
    dataset['Body Height [cm]'] = dataset['Body Height [cm]'] / 100
    # dataset.pop('Body Height [cm]')
    return dataset


def main():
    # Loading in training dataset using pandas
    dataset = pd.read_csv('tcd-ml-1920-group-income-train.csv')

    # Split dataset into target(y) and predictor variables(train)
    train = dataset
    #correlation = train.corr(method='pearson')
    #columns = correlation.nlargest(10, 'Total Yearly Income [EUR]').index
    #print(columns)

    # When data gets messed up
    train = train[:1044560]
    # Remove Duplicates
    train = train.drop_duplicates(subset='Instance', keep='first',
                                  inplace=False)

    # Fileter housing data
    # train = genderCleaning(train, False)
    #train = HousingSituation(train, False)
    train = processAdditionToSalary(train)
    y = train['Salary']
    # Put Income on Log Scale
    #train["Total Yearly Income [EUR]"] = train["Total Yearly Income [EUR]"].apply(np.log)
    #y = train['Total Yearly Income [EUR]'].values



    # Dropped columns
    train = train.drop(columns=['Instance',
                                'Hair Color',
                                'Crime Level in the City of Employement',
                                #'Satisfation with employer',
                                'Total Yearly Income [EUR]',
                                'Salary',
                                ])
    train = bodyHeight(train)
    train = changeSizeOfCity(train)
    train = degree(train)
    ct = imputation(train)
    regressor = CatBoostRegressor()

    X_train, X_test, y_train, y_test = train_test_split(
        train, y, test_size=0.2)


    additionSal = X_test['Yearly Income in addition to Salary (e.g. Rental Income)']
    #Want to drop the above column in X test and X train

    print ('Fitting')
    ml_pipe = Pipeline([
        ('transform', ct),
        ('regressor', regressor)])
    ml_pipe.fit(X_train, y_train)
    print ('Predicting')
    y_pred = ml_pipe.predict(X_test)
    print('MAE is: {}'.format(mean_absolute_error(y_test + additionSal, y_pred+additionSal)))
    #print('MAE is: {}'.format(mean_absolute_error(np.exp(y_test), np.exp(y_pred))))

    test_dataset = pd.read_csv(
        'tcd-ml-1920-group-income-test.csv')

    # Split into target and predictor variables
    predict_X = test_dataset
    # predict_X = genderCleaning(predict_X, True)
    #predict_X = HousingSituation(predict_X,True)
    predict_X = processAdditionToSalary(predict_X)
    additionSal = predict_X['Yearly Income in addition to Salary (e.g. Rental Income)']
    predict_X['Salary'] = predict_X['Total Yearly Income [EUR]'] - predict_X['Yearly Income in addition to Salary (e.g. Rental Income)']
    predict_y = predict_X['Salary']



    predict_X = predict_X.drop(columns=['Instance',
                                            'Hair Color',
                                            'Crime Level in the City of Employement',
                                            #'Yearly Income in addition to Salary (e.g. Rental Income)',
                                            'Salary',
                                            'Total Yearly Income [EUR]',
                                            #'Satisfation with employer',
                                            #'Housing Situation'
                                            ])
    predict_X = bodyHeight(predict_X)
    predict_X = changeSizeOfCity(predict_X)
    predict_X = degree(predict_X)

    predict_X['Work Experience in Current Job [years]'].replace('#NUM!', np.nan, inplace = True)

    # Predict using submission data
    pred2 = ml_pipe.predict(predict_X)
    print(pred2)
    # Write to file
    pred2 = pred2 + additionSal
    output = pd.read_csv('tcd-ml-1920-group-income-submission.csv')
    instance = output['Instance']
    output.pop('Instance')
    a = pd.DataFrame.from_dict({
        'Instance': instance,
        'Total Yearly Income [EUR]': pred2
    })
    a.to_csv("tcd-ml-1920-group-income-submission.csv", index=False)



if __name__ == "__main__":
    main()
