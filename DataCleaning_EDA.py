import os
os.chdir("C:\\Users\\paulo\\OneDrive - Universidade de Coimbra\\Ambiente de Trabalho\\Cursos\\Curso DataScience\\49. Projeto Prático Final\\Load_To_GitHub_AfterFinish")

from DataScience_Project import df
import pandas as pd
import numpy as np
import statistics as stats
import matplotlib.pyplot as plt
import seaborn as sns
#The goal is to lower the risk of defaulting clients, so we will use the data to predict which clients are more likely to fail to pay their loans.
#For this case, and having in mind that accuracy is not the only metric that we should use, for experimental porpuses we will use the accuracy score to evaluate the "goal" that we want to achieve.
#The goal regarding model performance is to achieve an accuracy score of 0.75 or higher.
#Of course that in a real-world scenario we would also consider with caution metrics like F1 score, Precision, Recall, etc.

# 0 - The client DID NOT PAY the loan
# 1 - The client PAID the loan

#DATA CLEANING and EXPLORATORY DATA ANALYSIS

#####################################################################################
#Initial Data Exploration
#####################################################################################
#print(df.head())
#print(df.shape) #Check STATUS: OK# - (1000, 20) - The dataframe has 1000 rows and 20 columns
#Check STATUS: OK#
#The dataframe has 1000 rows and 21 columns - DF sucessfully created from the database
#print(df.describe()) # Summary statistics for numerical features
#Check STATUS: OK#
#####################################################################################
#####################################################################################

#####################################################################################
#Check for missing values - Null values
#####################################################################################
print ("NULL VALUES: ", df.isnull().sum())
#Emprego - 10 missing values
###[Info]The data in "Emprego" is numerical encoded, but its a categorical feature. 
##[Decision] WE HAVE TWO OPTIONS - we can remove the lines with missing values or replace the missing values with the mode of the column
print ("MODE_EMPREGO: ", stats.mode(df["Emprego"])) #Mode = 3 
#ResidenciaDesde - 7 missing values
##[Decision] WE HAVE TWO OPTIONS - we can remove the lines with missing values or replace the missing values with the mode of the column
#Min.Value = 1
#Max.Value = 4
#How much time the client has been living in the current residence (years)
print ("MODE_RESIDENCIADESDE: ", stats.mode(df["ResidenciaDesde"])) #Mode = 4
#Habitacao - 9 missing values
###[Info]The data in "Habitacao" is numerical encoded, but its a categorical feature.
##[Decision] WE HAVE TWO OPTIONS - we can remove the lines with missing values or replace the missing values with the mode of the column
print ("MODE_HABITACAO: ", stats.mode(df["Habitacao"])) #Mode = 1

#Decision: Since the missing values are a small percentage of the total data, both remove or imputation is acceptable.
#We will use imputation to replace the missing values with the mode of the column, since it will allows us to retain the full dataset and at worse we will have a small bias in the data.
#Fill NA values
df["Emprego"] = df["Emprego"].fillna(3) #Replace Na with 3 
df["ResidenciaDesde"] = df["ResidenciaDesde"].fillna(4) #Replace Na with 4 and inplace = True to change the original dataframe.
df["Habitacao"] = df["Habitacao"].fillna(1) #Replace Na with 1 and inplace = True to change the original dataframe.
print (df.describe())
print ("NA VALUES: ", df.isna().sum()) #Check if the missing values were replaced
#STATUS: OK# - Missing values were replaced with the mode of the column
#####################################################################################
#####################################################################################

#####################################################################################
#Check for Duplicates and remove the column "IDCREDITO"
#####################################################################################
#Note: All columns can have duplicates, besides the ID column
print ("Duplicated Values:", df["IDCREDITO"].duplicated().sum()) #Check for duplicates
#OK NO DUPLICATES Found IN ["IDCREDITO"]
#REMOVE THE COLUMN "IDCREDITO" - It is not a feature that will help us to predict the goal variable
df = df.drop("IDCREDITO", axis = 1)
#####################################################################################
#####################################################################################

#####################################################################################
#Now we will...
#Use visualization to identify outliers
#We will use boxplots to identify outliers in the numerical features and chategorical features
#We will use histograms to check the distribution of the numerical features
#We will use value_counts to check the distribution of the categorical features
################### ADITIONAL INFORMATION ############################################
#Numerical Features = Duracao, Valor, Idade, Dependentes, 
#Categoriacl Features = HistoricoCredito, Proposito, Investimentos, Emprego, EstadoCivil, FiadorTerceiros, ResidenciaDesde, OutrosFinanciamentos, Habitacao, EmprestimoExistente, Profissao, SocioEmpresa, Estrangeiro, Status
#Although they are all "numerical" some of them are encoded as numerical but are categorical features or represent a category/range of values and so they should be treated as categorical features.
#####################################################################################
#####################################################################################


#####################################################################################
#Distribution study with Histograms - Numerical Features
#####################################################################################
fig, axs = plt.subplots(2, 2, figsize=(15, 10))  # 2 rows and 2 columns
fig.suptitle('Distribution Study - Numerical Features', fontsize=16)
sns.histplot(df["Duracao"], kde=True, ax=axs[0, 0])
#Status ok - Multimodal distribution    (4 - 72) lets consider it to be months

sns.histplot(df["Valor"], kde=True, ax=axs[0, 1])
#Status ok - Right-Skewed Distribution (250 - 18424) lets consider it to be dolars

sns.histplot(df["Idade"], kde=True, ax=axs[1, 0])
#Status ok - Right-Skewed Distribution (19-70) lets consider it to be  years

sns.histplot(df["Dependentes"], kde=True, ax=axs[1, 1])
#Status ok - Bimodal or Binary Distribution (1-2) lets consider it to be a binary value

# Adjust layout to prevent overlap
plt.tight_layout()
# Show the plot
plt.show()
#####################################################################################
#####################################################################################


#####################################################################################
#Study of Outliers Boxplots FOR NUMERICAL FEATURES
#####################################################################################
fig, axs = plt.subplots(2,2, figsize=(15, 10))  # 7 rows and 2 columns
fig.suptitle('Outlier Study for Numerical Features', fontsize=16)

df.boxplot(column = ["Duracao"], ax = axs[0,0])
#Status: OK# - We have "outliers" but the values are valid/make sense

df.boxplot(column = ["Valor"], ax = axs[0,1])
#Status: OK# - We have "outliers" but the values are valid/make sense

df.boxplot(column = ["Idade"], ax = axs[1,0])
#Status: OK# - We have "outliers" but the values are valid/make sense

df.boxplot(column = ["Dependentes"], ax = axs[1,1])
#Status: OK# - We have "outliers" but the values are valid/make sense (WE only have 2 values)

# Adjust layout to prevent overlap
plt.tight_layout()
# Show the plot
plt.show()
#####################################################################################
#####################################################################################

#####################################################################################
#Study of Destribution for Categorical Features
#####################################################################################
HistoricoCredito_counts = df["HistoricoCredito"].value_counts()
print (HistoricoCredito_counts)
#Status: OK# - 6 categories - Checks with SQL query

Proposito_counts = df["Proposito"].value_counts()
print (Proposito_counts)
#Status: OK# - 10 categories - Checks with SQL query

Investimentos_counts = df["Investimentos"].value_counts()
print (Investimentos_counts)
#Status: OK# - 3 categories - Checks with SQL query

Emprego_counts = df["Emprego"].value_counts()
print (Emprego_counts)
#Status: OK# - 5 categories - Checks with SQL query

TempoParcelamento_counts = df["TempoParcelamento"].value_counts()
print (TempoParcelamento_counts)
#Status: OK# - 4 categories - Checks with SQL query

EstadoCivil_counts = df["EstadoCivil"].value_counts()
print (EstadoCivil_counts)
#Status: OK# - 4 categories - Checks with SQL query

FiadorTerceiros_counts = df["FiadorTerceiros"].value_counts()
print (FiadorTerceiros_counts)
#Status: OK# - 4 categories - Checks with SQL query

ResidenciaDesde_counts = df["ResidenciaDesde"].value_counts()
print (ResidenciaDesde_counts)
#Status: OK# - 4 categories - Checks with SQL query

OutrosFinanciamentos_counts = df["OutrosFinanciamentos"].value_counts()
print (OutrosFinanciamentos_counts)
#Status: OK# - 3 categories - Checks with SQL query

Habitacao_counts = df["Habitacao"].value_counts()
print (Habitacao_counts)
#Status: OK# - 3 categories - Checks with SQL query

EmprestimoExistente_counts = df["EmprestimoExistente"].value_counts()
print (EmprestimoExistente_counts)
#Status: OK# - 4 valid categories(1 category is a outlier -> "999") - Checks with SQL query

Profissao_counts = df["Profissao"].value_counts()
print (Profissao_counts)
#Status: OK# - 4 categories - Checks with SQL query

SocioEmpresa_counts = df["SocioEmpresa"].value_counts()
print (SocioEmpresa_counts)
#Status: OK# - 2 categories - Checks with SQL query

Estrangeiro_counts = df["Estrangeiro"].value_counts()
print (Estrangeiro_counts)
#Status: OK# - 2 categories - Checks with SQL query
#####################################################################################
#####################################################################################

#####################################################################################
#Study of Outliers Boxplots for Categorical Features
#The use of boxplots with categorical data have limitations, and boxplot analysis needs to be done with caution.
#For example, the boxplot can show/identify "outliers" that are valid values, so we need to be careful when interpreting the boxplot.
#####################################################################################

fig, axs = plt.subplots(2,7, figsize=(10, 10))  # 7 rows and 2 columns
fig.suptitle('Outlier Study for Categorical Features', fontsize=16)

df.boxplot(column = ["HistoricoCredito"], ax = axs[0,0])
#Status: It shows "outliers" but the outliers are valid values

df.boxplot(column = ["Proposito"], ax = axs[0,1])
#Status: OK#

df.boxplot(column = ["Investimentos"], ax = axs[0,2])
#Status: OK#

df.boxplot(column = ["Emprego"], ax = axs[0,3])
#Status: OK - Have some outliers but they are valid values

df.boxplot(column = ["TempoParcelamento"], ax = axs[0,4])
#Status: OK 

df.boxplot(column = ["EstadoCivil"], ax = axs[0,5])
#Status: OK 

df.boxplot(column = ["FiadorTerceiros"], ax = axs[0,6])
#Status: OK - Have some outliers but they are valid values

df.boxplot(column = ["ResidenciaDesde"], ax = axs[1,0])
#Status: OK 

df.boxplot(column = ["OutrosFinanciamentos"], ax = axs[1,1])
#Status: OK - Have some outliers but they are valid values

df.boxplot(column = ["Habitacao"], ax = axs[1,2])
#Status: OK 

df.boxplot (column = ["EmprestimoExistente"], ax = axs[1,3])
#Status: OK - Have some outliers but they are valid values

df.boxplot(column = ["Profissao"], ax = axs[1,4])
#Status: Not-OK - 1 WE HAVE one outlier - substitute with mode

df.boxplot(column = ["SocioEmpresa"], ax = axs[1,5])
#Status: OK# 0-1 binary variable

df.boxplot(column = ["Estrangeiro"], ax = axs[1,6])
#Status: OK# 0-1 binary variable

# Adjust layout to prevent overlap
plt.tight_layout()
# Show the plot
plt.show()
#####################################################################################
#####################################################################################


#####################################################################################
#TREATMENT OF OUTLIERS - NUMERICAL AND CATEGORICAL FEATURES
#####################################################################################
#NOTE!
#Outliers where identified in the column "Profissao" - We will replace the outlier with the mode of the column


#print ("Mode Profissao:", stats.mode(df["Profissao"])) #Mode = 4
mode_profissao = stats.mode(df["Profissao"]) 
mean_profissao = df["Profissao"].mean() #12.273
std_profissao = df["Profissao"].std() #94.0863
threshold = 3 # 3 standard deviations
#print (mean_profissao, std_profissao)
#Identify Outliers
#We define that the outliers are the values that are 3 standard deviations away from the mean
#We knoe that we have outliers in the column "Profissao" because the boxplot shows one outlier, above the upper whisker.
#So we will only identify the outliers that are "> mean + 3*std"
outliers = (df["Profissao"] > mean_profissao + (threshold*std_profissao))
df.loc [outliers, "Profissao"] = mode_profissao
df.boxplot(column = ["Profissao"])
plt.show()
#Status ok - the outlier was replaced with the mode of the column - We still have outlier but they are valid values
#####################################################################################
#####################################################################################


#####################################################################################
#STUDY OF THE GOAL VARIABLE ####################Goal Variable = Status###############
#####################################################################################
df.boxplot(column = ["Status"])
plt.title = ("STATUS")
plt.show() #Status: OK# 0-1 binary variable/Goal variable
#######################Goal Variable#######################

#Check the distribution of the goal variable
status_counts = df["Status"].value_counts()
print (status_counts)
#WE HAVE AN IMBALANCED DATASET
#The dataset has 700 clients that paid their loans and 300 clients that failed to pay their loans
#We will use SMOTE to balance the dataset, we could also use other techniques like undersampling, oversampling, etc.

print (df.describe())
#####################################################################################
#EVERYTHING SEEMS OK - WE CAN MOVE TO THE NEXT STEP
