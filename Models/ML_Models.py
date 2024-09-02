#####################################################################################
#load the data from the DataCleaning_EDA.py
#####################################################################################
import os
os.chdir("C:\\Users\\paulo\\OneDrive - Universidade de Coimbra\\Ambiente de Trabalho\\Cursos\\Curso DataScience\\49. Projeto Prático Final\\Load_To_GitHub_AfterFinish")
from DataCleaning_EDA import df
#####################################################################################
#####################################################################################

#####################################################################################
#NOTE FOR Building the Machine Learning Model
#####################################################################################

#GOAL: Predict the STATUS OF CREDITO based on the other features in the dataset
#We want to know which client are more likely to pay their loans and which are more likely to fail to pay their loans
#The models  that we will use/test are: Logistic Regression, Random Forest, Gradient Boosting and XGBoost
#Then we will compare the models and choose the best one
#####################################################################################
#####################################################################################

#####################################################################################
#import the libraries
#####################################################################################
import matplotlib.pyplot as plt
import seaborn as sns
#import pandas as pd
#Data Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
#1º Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
#Evaluation/Metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#####################################################################################
#####################################################################################

#####################################################################################
#Split the data into features and target
#####################################################################################
x = df.drop("Status", axis = 1)
y = df["Status"]
#####################################################################################
#####################################################################################

#####################################################################################
#Although they are all "numerical" some of them are encoded as numerical but are categorical features or represent a category/range of values and so they should be treated as categorical features.
#Here will use onehotenconding to transform the categorical features into numerical features and we will use StandardScaler to scale the numerical features.
#Categoriacl Features = HistoricoCredito, Proposito, Investimentos, Emprego, EstadoCivil, FiadorTerceiros, ResidenciaDesde, OutrosFinanciamentos, Habitacao, EmprestimoExistente, Profissao, SocioEmpresa, Estrangeiro, Status
#Numerical Features = Duracao, Valor, Idade, Dependentes
#####################################################################################
categorical_features = ["HistoricoCredito", "Proposito", "Investimentos", "Emprego", "EstadoCivil", "FiadorTerceiros", "ResidenciaDesde", "OutrosFinanciamentos", "Habitacao", "EmprestimoExistente", "Profissao", "SocioEmpresa", "Estrangeiro"]
numerical_features = ["Duracao", "Valor", "Idade", "Dependentes"]

#Preprocessing categorical features
categorical_transformer = OneHotEncoder(drop = "first")
#Preprocessing numerical features
numerical_transformer = StandardScaler()


#APPLY THE PREPROCESSING
preprocessor = ColumnTransformer(
    transformers = [
        ("cat", categorical_transformer, categorical_features),
        ("num", numerical_transformer, numerical_features)
    ])
#####################################################################################
#We have two options, either we process it "manually" or we use a pipeline
#For the sake of readability we will use a pipeline, because we also want to test various models and this way we can easily change the model and keep the preprocessing the same.
#We will also use the SMOTE technique to balance the data.(We have a lot more "good" than "bad" loans)

#HERE IS A EXAMPLE OF PREPROCESSING THE DATA MANUALLY
# Fit the preprocessor on the training data
#preprocessor.fit(x)

x_train, x_test, y_train, y_test = train_test_split (x, y, test_size = 0.2, random_state=42)

# Transform the training and testing data
#x_train_transformed = preprocessor.transform(x_train)
#x_test_transformed = preprocessor.transform(x_test)

# Print the transformed data
#print(x_train_transformed)
#print(x_test_transformed)
#print(y_train)
#print(y_test)

#In the STATUS column:
#1 is good - paid
#0 is bad - not paid
#####################################################################################

#Create the Pipeline
#Apllly the SMOTE technique to balance the data appears to improve the results of the models
#So we will use the SMOTE technique to balance the data.
pipeline = ImbPipeline(
    steps =[("preprocessor", preprocessor),
            ("smote", SMOTE(random_state = 42)),
            ("classifier", LogisticRegression(random_state=42))
            ])

#Fit the model - Logistic Regression
model_logistic_regression = pipeline.fit(x_train, y_train)
#Predictions - Logistic Regression
pred_Logistic = pipeline.predict(x_test)

#Evaluate the model
print (confusion_matrix(y_test, pred_Logistic))
#                Predicted
#          Negative    Positive  True Positive (TP): Correctly predicted positive cases (84)    - PROFIT - PEOPLE WHO WOULD PAY AT WE GAVE THE LOAN
#Actual                          True Negative (TN): Correctly predicted negative cases (43)    - PROFIT - PEOPLE WHO WOULD NOT PAY AT WE DID NOT GIVE THE LOAN
#Negative  TN (43)     FP (18)   False Positive (FP): Incorrectly predicted positive cases (18) - LOSS - PEOPLE WHO WOULD NOT PAY AT WE GAVE THE LOAN
#Positive  FN (55)     TP (84)   False Negative (FN): Incorrectly predicted negative cases (55) - LOSS - PEOPLE WHO WOULD PAY AT WE DID NOT GIVE THE LOAN

#Plot the confusion matrix
cm= confusion_matrix(y_test, pred_Logistic)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt="d", cmap = "Blues", xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print (classification_report(y_test, pred_Logistic))
#              precision    recall  f1-score   support
#           0       0.44      0.70      0.54        61  Precision for class 0: 44% of the predicted bad loans were actually bad.
#           1       0.82      0.60      0.70       139  Recall for class 0: 70% of the actual bad loans were correctly predicted as bad.
#    accuracy                           0.64       200  Precision for class 1: 82% of the predicted good loans were actually good.
#   macro avg       0.63      0.65      0.62       200  Recall for class 1: 60% of the actual good loans were correctly predicted as good.
#weighted avg       0.71      0.64      0.65       200  The F1 score is the harmonic mean of precision and recall. 
#                                                       Support is the number of actual occurrences of the class in the dataset. For class 0, there are 61 actual bad loans.
#                                                       For class 1, there are 139 actual good loans.
print ("Accuracy Logistic Regression: ", accuracy_score(y_test, pred_Logistic))
# Accuracy: The overall accuracy of the model is 64%.





#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#Hyperparameter Tuning FOR Random Forest - Gradient Boosting - XGBoost
#####################################################################################
#TRY GridSearchCV or RandomizedSearchCV to find the best hyperparameters for the Random Forest and Gradient Boosting models
#GridSearchCV is more exhaustive and will try all the possible combinations of hyperparameters
#RandomizedSearchCV will try a random sample of the hyperparameters to a maximum number of iterations defined by the user
#####################################################################################

#Define the parameters

#Random Forest model Parameters to tune
#Create dictionary with hyperparameters to tune (Key-Valueus they can take)
param_grid_rf = {
                "n_estimators": [100, 200, 300],
                 "max_depth": [None, 10, 20, 30],
                 "min_samples_split": [2, 5, 10],
                 "min_samples_leaf": [1, 2 ,4 ],
                 "max_features": ["sqrt", "log2", None],
                 "bootstrap" : [True, False]  
}
#Gradient boosting model Parameters to tune
#Create dictionary with hyperparameters to tune (Key-Valueus they can take)
param_grid_gb = {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "subsample": [0.8, 0.9, 1.0],
                "max_features": ["sqrt", "log2", None],
                "loss": ["log_loss", "exponential"]
}
# XGBoost model Parameters to tune
# Define the hyperparameter grid
param_grid_xgb = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_child_weight': [1, 3, 5],
                'gamma': [0, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.01, 0.1],
                'reg_lambda': [1, 1.5, 2]
}
#Get the best hyperparameters for the Random Forest model and the Gradient Boosting model
#rf = RandomForestClassifier(random_state=42)
#gb = GradientBoostingClassifier(random_state=42)
#xgb = xgb.XGBClassifier(random_state=42)

#Perform the RandomSearch #MAXIMUM of maximum ITERATIONS IS 648
#random_search_rf = RandomizedSearchCV(estimator=rf, param_distributions=param_grid_rf, n_iter= 300, cv=5, n_jobs=-1, verbose=2, random_state=42)
#random_search_gb = RandomizedSearchCV(estimator=gb, param_distributions=param_grid_gb, n_iter= 300, cv=5, n_jobs=-1, verbose=2, random_state=42)
#random_search_xgb = RandomizedSearchCV(estimator=xgb, param_distributions=param_grid_xgb, n_iter= 300, cv=5, n_jobs=-1, verbose=2, random_state=42)

#Perform the GridSearchCV #MAXIMUM NUMBER OF ITERATIONS IS 648
#grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=2)
#grid_search_gb = GridSearchCV(estimator=gb, param_grid=param_grid_gb, cv=5, n_jobs=-1, verbose=2)
#grid_search_xgb = GridSearchCV(estimator=xgb, param_grid=param_grid_xgb, cv=5, n_jobs=-1, verbose=2)

#Fit the Data in to the RandomSearchCV
#random_search_rf.fit(x_train, y_train)
#random_search_gb.fit(x_train, y_train)
#random_search_xgb.fit(x_train, y_train)

#Fit the Data in to the GridSearchCV
#grid_search_rf.fit(x_train, y_train)
#grid_search_gb.fit(x_train, y_train)
#grid_search_xgb.fit(x_train, y_train)

#Get the best hyperparameters
#####################################################################################
##################################RANDOM FOREST######################################
#####################################################################################
#RandomSearchCV
#print("Best Hyperparameters for RF - Using RandomSearch: ", random_search_rf.best_params_)
#print("Best Score for RF - Using RandomSearch: ", random_search_rf.best_score_)
#GridSearchCV
#print ("Best Hyperparameters for RF - Using GridSearch: ", grid_search_rf.best_params_)
#print ("Best Score for RF - Using GridSearch: ", grid_search_rf.best_score_)
#####################################################################################
###################################Gradient Boost####################################
#####################################################################################
#RandomSearchCV
#print("Best Hyperparameters for GB - Using RandomSearch: ", random_search_gb.best_params_)
#print("Best Score for GB - Using RandomSearch: ", random_search_gb.best_score_)
#GridSearchCV
#print ("Best Hyperparameters for GB - Using GridSearch: ", grid_search_gb.best_params_)
#print("Best Score for GB - Using GridSearch: ", grid_search_gb.best_score_)
#####################################################################################
################################XGB-BOOST############################################
#####################################################################################
#RandomSearchCV
#print("Best Hyperparameters for XGB - Using RandomSearch: ", random_search_xgb.best_params_)
#print("Best Score for XGB - Using RandomSearch: ", random_search_xgb.best_score_)
#GridSearchCV
#print ("Best Hyperparameters for XGB - Using GridSearch: ", grid_search_xgb.best_params_)
#print("Best Score for XGB - Using GridSearch: ", grid_search_xgb.best_score_)
#####################################################################################
######################CONCLUSION-CONCLUSION-CONCLUSION###############################
#####################################################################################
#Best Hyperparameters for RF: {'n_estimators': 200, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_features': None, 'max_depth': None, 'bootstrap': True}
#Best Score for RF: 0.7725000000000001
#Best Hyperparameters for GB: {'subsample': 1.0, 'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 3, 'loss': 'log_loss', 'learning_rate': 0.1}
#Best Score for GB: 0.7649999999999999
#Best Hyperparameters for XGB - Using GridSearch:  {'colsample_bytree': 0.9, 'gamma': 0, 'learning_rate': 0.01, 'max_depth': 7, 'min_child_weight': 3, 'n_estimators': 300, 'reg_alpha': 0, 'reg_lambda': 1, 'subsample': 0.8}
#Best Score for XGB - Using GridSearch:  0.7625

#####################################################################################
#Note: The code above is commented because it takes to long to run and the results are already known - Check Conclusion
#We will use GridSearchCV because the number of iterations is relatively small, 648, and we can afford to run it.
#In the case of lack of computer power we could use RandomizedSearchCV, but we would have to increase the number of iterations and risk not finding the true best hyperparameters.
#We will use the best hyperparameters obtained to build the models.
#The code works, if you want to test it, just uncomment it.
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################




#####################################################################################
#APPLY THE BEST HYPERPARAMETERS TO THE MODELS
#####################################################################################


#2º TUNNEED RANDOM FOREST
RF_BEST = RandomForestClassifier(n_estimators= 200, min_samples_split= 10, min_samples_leaf=1, max_features=None, max_depth= None, bootstrap= True, random_state=42)
# Create pipelines for both models
pipeline2 = ImbPipeline(
    steps =[("preprocessor", preprocessor),
            ("smote", SMOTE(random_state = 42)),
            ("classifier", RF_BEST)
            ])
#Fit the model with the best hyperparameters
model_RF_BEST = pipeline2.fit(x_train, y_train)
#make predictions
pred_RF_BEST = pipeline2.predict(x_test)
#Evaluate the model
cm = confusion_matrix(y_test, pred_RF_BEST)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt="d", cmap = "Blues", xticklabels=["Predicted Negative", "Predicted Positive"], yticklabels=["Actual Negative", "Actual Positive"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
#Classification Report
print (classification_report(y_test, pred_RF_BEST))
#Accuracy
print ("Accuracy - RandomForest-BEST: ", accuracy_score(y_test, pred_RF_BEST))
print (confusion_matrix(y_test, pred_RF_BEST))
#####################################################################################


#3º TUNNEED GRADIENT BOOSTING CLASSIFIER
GB_BEST = GradientBoostingClassifier(subsample= 1.0, n_estimators= 100, min_samples_split= 10, min_samples_leaf= 1, max_features= "sqrt", max_depth= 3, loss= "log_loss", learning_rate= 0.1, random_state=42)
# Create pipelines for both models
pipeline3 = ImbPipeline(
    steps =[("preprocessor", preprocessor),
            ("smote", SMOTE(random_state = 42)),
            ("classifier", GB_BEST)
            ])
#Fit the model with the best hyperparameters
model_GB_BEST = pipeline3.fit(x_train, y_train)
#Make predictions
pred_GB_BEST = pipeline3.predict(x_test)
#Evaluate the model
cm = confusion_matrix(y_test, pred_GB_BEST)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt="d", cmap ="Blues", xticklabels=["Predicted Negative", "Predicted Positive"], yticklabels=["Actual Negative", "Actual Positive"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
#Classification Report
print (classification_report(y_test, pred_GB_BEST))
#Accuracy
print ("Accuracy - GradientBoosting-BEST: ", accuracy_score(y_test, pred_GB_BEST))
print (confusion_matrix(y_test, pred_GB_BEST))
#####################################################################################


#4º TUNNEED XGBoost
XGB_BEST = xgb.XGBClassifier(colsample_bytree= 0.9, gamma= 0, learning_rate= 0.01, max_depth= 7, min_child_weight= 3, n_estimators= 300, reg_alpha= 0, reg_lambda= 1, subsample= 0.8, random_state=42)
# Create pipelines for both models
pipeline4 =  ImbPipeline(steps= 
                         [("preprocessor", preprocessor),
                          ("smote", SMOTE(random_state = 42)),
                          ("classifier", XGB_BEST)
                         ])
#Fit the model
model_XGB_BEST = pipeline4.fit (x_train, y_train)
#Make predictions
pred_XGB = pipeline4.predict(x_test)
#Evaluate the model
cm = confusion_matrix(y_test, pred_XGB)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt="d", cmap ="Blues", xticklabels=["Predicted Negative", "Predicted Positive"], yticklabels=["Actual Negative", "Actual Positive"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
#Classification Report
print (classification_report(y_test, pred_XGB))
#Accuracy
print ("Accuracy - XGBoost-BEST: ", accuracy_score(y_test, pred_XGB))
print (confusion_matrix(y_test, pred_XGB))
#####################################################################################
#SUMMARY RESULTS

#Model Evaluation
#1. Logistic Regression
#Class 0 (Bad Payer): Precision: 0.44, Recall: 0.70, F1-score: 0.54
#Class 1 (Good Payer): Precision: 0.82, Recall: 0.60, F1-score: 0.70
#Accuracy: 0.635
#2. RandomForest-BEST
#Class 0 (Bad Payer): Precision: 0.54, Recall: 0.59, F1-score: 0.56
#Class 1 (Good Payer): Precision: 0.81, Recall: 0.78, F1-score: 0.79
#Accuracy: 0.72
#Confusion Matrix:
#Class 0: [36 True Negatives, 25 False Positives]
#Class 1: [31 False Negatives, 108 True Positives]
#3. GradientBoosting-BEST
#Class 0 (Bad Payer): Precision: 0.56, Recall: 0.61, F1-score: 0.58
#Class 1 (Good Payer): Precision: 0.82, Recall: 0.79, F1-score: 0.81
#Accuracy: 0.735
#Confusion Matrix:
#Class 0: [37 True Negatives, 24 False Positives]
#Class 1: [29 False Negatives, 110 True Positives]
#4. XGBoost-BEST
#Class 0 (Bad Payer): Precision: 0.59, Recall: 0.43, F1-score: 0.50
#Class 1 (Good Payer): Precision: 0.78, Recall: 0.87, F1-score: 0.82
#Accuracy: 0.735
#Confusion Matrix:
#Class 0: [26 True Negatives, 35 False Positives]
#Class 1: [18 False Negatives, 121 True Positives]

#####################Conclusion####################################################

#Precision for Class 0 (Bad Payers): The XGBoost-BEST model has the highest precision (0.59), meaning fewer bad payers are misclassified as good payers. However, its recall for bad payers is lower (0.43).

#Recall for Class 0 (Bad Payers): The GradientBoosting-BEST model has a higher recall (0.61), meaning it captures more of the bad payers than XGBoost-BEST.

#F1-Score for Class 0 (Bad Payers): The GradientBoosting-BEST model also has the highest F1-score (0.58) for class 0, which balances precision and recall effectively.

#Class 1 (Good Payers): All models have good performance, but XGBoost-BEST has the highest F1-score (0.82) and recall (0.87), indicating strong performance in identifying good payers.

#Accuracy: Both GradientBoosting-BEST and XGBoost-BEST have the highest accuracy at 0.735, but accuracy alone isn't enough given the importance of identifying bad payers accurately.

#Best Model: GradientBoosting-BEST
#While XGBoost-BEST has strong metrics for good payers, GradientBoosting-BEST offers the best balance across all metrics, particularly for the critical class 0 (bad payers). Its higher recall and F1-score for bad payers make it the most reliable model for minimizing the risk of giving loans to bad payers while still effectively identifying good payers.

