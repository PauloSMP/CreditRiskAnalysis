# Here we will try to implement a Neural Network model using Keras
#Objective: Risk analysis - to predict the risk of a person not paying back the loan
# we want to get more than:

#3. GradientBoosting-BEST
#Class 0 (Bad Payer): Precision: 0.56, Recall: 0.61, F1-score: 0.58
#Class 1 (Good Payer): Precision: 0.82, Recall: 0.79, F1-score: 0.81
#Accuracy: 0.735
#Confusion Matrix:
#Class 0: [37 True Negatives, 24 False Positives]
#Class 1: [29 False Negatives, 110 True Positives]


#####################################################################################
import os
os.chdir("C:\\Users\\paulo\\OneDrive - Universidade de Coimbra\\Ambiente de Trabalho\\Cursos\\Curso DataScience\\49. Projeto Prático Final\\Load_To_GitHub_AfterFinish")
from DataCleaning_EDA import df
#####################################################################################
#Importing the libraries
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras_tuner as kt
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE


#####################################################################################
#Split the data into features and target
#####################################################################################
x = df.drop("Status", axis = 1)
y = df["Status"]
#####################################################################################
#Split the data into training and testing
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

x_train, x_test, y_train, y_test = train_test_split (x, y, test_size = 0.3, random_state=42)

#print (x_train.shape)   #(700, 18)
#print (x_test.shape)    #(300, 18)
#print (x_train.head())

x_train = preprocessor.fit_transform(x_train)
x_test = preprocessor.transform(x_test)

#Apply the SMOTE technique to balance the data
smote = SMOTE(random_state=42)
x_train, y_train = smote.fit_resample(x_train, y_train)

#print (x_train.shape)   #(980, 47)
#print (x_train)
#print (x_test.shape)    #(300, 47)
#The shape of the data changes due to preprocessing and SMOTE.
#The data is now ready to be used in the model, store in a sparse matrix format.

def build_model(hp):
    model = Sequential ()
    
    #Define number of layers
    num_layers = hp.Int("num_layers", min_value=1, max_value=5, step = 1)
    
    #Input Layer
    model.add(Dense(units=hp.Int("units_input", min_value=32, max_value=512, step=8), activation="relu", input_shape=(x_train.shape[1],)))
    model.add(Dropout(hp.Float("dropout_input", min_value=0.1, max_value=0.5, step= 0.1)))
    
    #Hidden Layers
    for i in range(num_layers):
        model.add(Dense(units=hp.Int("units_" + str(i), min_value=32, max_value=512, step=8), activation =hp.Choice("activation_" + str(i), ["relu", "tanh", "sigmoid", "softmax", "linear"])))
        model.add(Dropout(hp.Float("dropout_" + str(i), min_value=0.1, max_value=0.5, step=0.1)))
      
        #Output Layer
    model.add(Dense(units=1, activation="sigmoid"))
    
    #Compile
    model.compile(optimizer=hp.Choice("optimizer",["adam", "rmsprop", "sgd", "adamw", "nadam"]), loss="binary_crossentropy", metrics=["accuracy"])
    return model

#Initiate the Tuner
import os
dir_path = r"C:\Temp\Tuner"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

tuner = kt.Hyperband(
    build_model,
    objective = "val_loss",
    max_epochs = 100,
    factor = 3,
    directory = dir_path,
    project_name = "CreditRisk-HP-Search"
)
 
#Early Stopping
early_stopping = EarlyStopping(monitor="val_loss", patience = 10, restore_best_weights = True)

tuner.search (x_train, y_train, epochs=100, validation_data=(x_test, y_test), callbacks=[early_stopping])

###END OF THE TUNING PROCESS###The best model is now stored in the tuner object

#Get the best model
best_model = tuner.get_best_models(num_models=1)[0] #Get the best model stored in the position 0
#The models are stored in a list, and sorted according to the performance metric defined in the tuner
#So this line will retrieve 1 model(num_models=1), being the best one [0]

# Print the model summary
print("Best Model Summary:")
best_model.summary()

# Save the best model
best_model.save("best_model_NEURAL_NETWORK.keras")

#Extract and print the configuration of the best model
print ("Best Model Configuration:")
for layer in best_model.layers:
    print (layer.get_config())

#we can now evaluate the model
test_loss, test_accuracy = best_model.evaluate(x_test, y_test)
print ("Test Loss - BEST MODEL: ", test_loss)
print ("Test Accuracy - BEST MODEL: ", test_accuracy)

#Make predictions
y_pred = (best_model.predict(x_test) > 0.5).astype("int32") #Give results in 0 or 1 insted of boolean True or False

#Evaluate the model - metrics
print ("Classification Report - BEST MODEL: ")
print (classification_report(y_test, y_pred))

#Confusion Matrix
print ("Confusion Matrix - BEST MODEL: ")
cm = confusion_matrix(y_test, y_pred)
print (cm)
#Calculate the accuracy
print ("Accuracy - BEST MODEL: ")
accuracy = accuracy_score(y_test, y_pred)
print (accuracy)

#RESULTS
#Classification Report - BEST MODEL: 
#              precision    recall  f1-score   support
#           0       0.65      0.53      0.59        90
#           1       0.81      0.88      0.84       210
#    accuracy                           0.77       300
#   macro avg       0.73      0.70      0.71       300
#weighted avg       0.76      0.77      0.77       300
#
#Confusion Matrix - BEST MODEL:
#[[ 48  42]
# [ 26 184]]
#Accuracy - BEST MODEL:
#0.7733333333333333

#Conclusion:

#we achieved a better model compared to the previous XGBoost model (accuracy: 0.735)
#Also the aim of the project was achieved, because we got more than 0.75 of accuracy

#Further hyperparameter tuning could be done to improve the model

#Also regarding the data, we could try to get more data to improve the model
#Especialy for the class 0, which has a lower precision and recall compared to class 1, and is the class we are most interested in predicting correctly
#We have a very higher number of samples for class 1 compared to class 0, so we could try to get more samples for class 0