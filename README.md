## tcd-group-competition-team13
#ML Assignment

#Group 13:
1. B60F41293D4354841FC0AA3B50638DAF1146AB15 - (Bryan Tyrrell)
2. C21F931B43563D95AB873D45556D6C285F9FE6BC - (Bronagh Carolan)
3. D0DF33C66A1E7CA124BCF64124DA95499318761F - (Michael Cullen)

#Final position:
17th - 9862.38936 MAE

# Models tested:
Squential neural network
XGBoost
CatBoost
Linear Regression

# Final model used:
CatBoost

# Preprocessing Steps:
1. Log Size of City and only include data less than 3000
2. Work Experience - Insert #NUM! with number 1, convert to float array and center around it's mean
3. Additional Salary - remove "EUR" and convert to float array
4. Degree - inserted "Missing" for missing values
5. Gender - Replaced F with female, convert 0 and nan to unknown and replaced missing values based on body Height
6. Used bfill and ordinally encoded
7. Body Height - scaled around the mean
8. Profession - take first 5 characters and bfill missing data
9. Crime - Scale around mean
10. Year - Based year on Housing Situation

# Encoding:
Started with getDummies encoding but ended up with TargetEncoding

# Regression Steps:
We used RandomSearchCV to get the best HyperParameters for the catboost regression mainly:
1. iterations
2. depth
3. learning_rate
We used an eval dataset to stop overfitting to make sure that our model did not overfit and we used od_type='IncToDec' which is catboost's inbuilt overfitting detection.


# Final Submission:
regressionCatBoostFinalScript.py
