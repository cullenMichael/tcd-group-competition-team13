import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from category_encoders import *
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, MinMaxScaler,
                                   OrdinalEncoder, StandardScaler)


def imputation(train):
    train['Work Experience in Current Job [years]'].replace('#NUM!', np.nan, inplace= True)

    train.pop('Instance')
    train.pop('Yearly Income in addition to Salary (e.g. Rental Income)')

    ord_si_step = ('si', SimpleImputer(strategy='constant',
                                       fill_value='MISSING'))
    ord_oe_step = ('oe', OrdinalEncoder())
    ord_steps = [ord_si_step, ord_oe_step]
    ord_pipe = Pipeline(ord_steps)
    ord_cols = ['University Degree', 'Satisfation with employer']

    cat_si_step = ('si', SimpleImputer(strategy='constant',
                                       fill_value='MISSING'))
    cat_ohe_step = ('te', TargetEncoder())
    cat_steps = [cat_si_step, cat_ohe_step]
    cat_pipe = Pipeline(cat_steps)
    cat_cols = ['Gender', 'Country', 'Profession','Hair Color', 'Housing Situation']

    num_cols = ['Year of Record', 'Age', 'Body Height [cm]', 'Crime Level in the City of Employement', 'Work Experience in Current Job [years]']
    num_si_step = ('si', SimpleImputer(strategy='median'))
    num_ss_step = ('ss', StandardScaler())
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


def main():
    # Loading in training dataset using pandas
    dataset = pd.read_csv(
        'tcd-ml-1920-group-income-train.csv')

    # Split dataset into target(y) and predictor variables(train)
    train = dataset
    y = train.pop('Total Yearly Income [EUR]').values
    ct = imputation(train)
    regressor = CatBoostRegressor()

    X_train, X_test, y_train, y_test = train_test_split(
        train, y, test_size=0.2)

    ml_pipe = Pipeline([
        ('transform', ct),
        ('regressor', regressor)])
    ml_pipe.fit(X_train, y_train)

    y_pred = ml_pipe.predict(X_test)
    print('MAE is: {}'.format(mean_absolute_error(y_test, y_pred)))

    test_dataset = pd.read_csv(
        'tcd-ml-1920-group-income-test.csv')

    # Split into target and predictor variables
    predict_X = test_dataset
    predict_y = predict_X.pop('Total Yearly Income [EUR]').values
    predict_X.pop('Yearly Income in addition to Salary (e.g. Rental Income)')
    predict_X.pop('Instance')

    predict_X['Work Experience in Current Job [years]'].replace('#NUM!', np.nan, inplace= True)

    # Predict using submission data
    pred2 = ml_pipe.predict(predict_X)
    print(pred2)
    # Write to file
    test = {'Total Yearly Income [EUR]': pred2}
    print (test)
    df_out = pd.DataFrame(test, columns=['Total Yearly Income [EUR]'])
    df_out.to_csv("tcd-ml-1920-group-income-submission.csv")


if __name__ == "__main__":
    main()
