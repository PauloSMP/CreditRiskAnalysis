
#We have diffrent ways to connect to a database, in this case we are going to use the psycopg2 library to connect to a PostgreSQL database.
#This option gives a warning message, but it works perfectly.
#The message: c:\Users\paulo\OneDrive - Universidade de Coimbra\Ambiente de Trabalho\Untitled-1.py:14: UserWarning: pandas only supports SQLAlchemy connectable (engine/connection) or database string URI or sqlite3 DBAPI2 connection. Other DBAPI2 objects are not tested. Please consider using SQLAlchemy."
#So we will use the SQLAlchemy library to connect to the database.

#USING PSYCOPG2
#import psycopg2
#import pandas as pd

#conn = psycopg2.connect (database = "credito",
#                         user = "postgres",
#                         host = "localhost",
#                         password = "2729",
#                         port = 5432)

#cur = conn.cursor()

#query = 'SELECT * FROM "CREDITO"'
#df = pd.read_sql_query(query, conn)        
#cur.close ()
#conn.close ()
#print (df.head())
#print (df.shape)

#USING SQLALCHEMY

from sqlalchemy import create_engine
import pandas as pd

#Creat a SQLAlchemy engine to connect to the database
#The conneciton string is in the format "postgresql+psycopg2://<username>:<password>@<host>:<port>/<database>"
engine = create_engine("postgresql+psycopg2://postgres:2729@localhost:5432/credito")

#Define a query to the database
query = 'SELECT * FROM "CREDITO"'

#Save the data from the query to a DataFrame
df = pd.read_sql_query(query,engine)

#Close the connection to the database
engine.dispose()

#NOTE: THE DATABASE HAVE SOME FLAWS. FOR EXAMPLE, THE "CLIENTES" TABLE DONT HAVE A MATCH TO THE CENTRAL "CREDITO" TABLE. SO WE WILL IGORE IT FOR NOW.
#WE HAVE A CENTRAL COLUMN THAT IS ENCODED NUMERICALLY BUT IT IS A CATEGORICAL FEATURE THAT HAVE A MATCH TO THE SECONDARY COLUMNS.
#FOR EXAMPLE, WE COULD USE " SELECT * FROM "CREDITO" FULL JOIN "EMPREGO" ON "CREDITO"."Emprego" = "EMPREGO"."IDEMPREGO";", in order to add ..
#the "EMPREGO" table to the "CREDITO" table. and insted of having 1,2,3,4,5 WE WOULD HAVE "<1"; ">=7"; "1<=X<4"; "4<=X<7"; "Desempregado".

#We will not waste time with this: we will: 
# use the encoded data and treat it as categorical data is they have a categorical meaning/match with another table.
# use the encoded data and treat it as numerical data if they dont have a categorical meaning/match with another table or if they dont have a match with another table.
#To use in the ML model we will encoded the categorical data using OneHotEncoding and we will scale the numerical data using StandardScaler. 
#This was we will achieve the same as if we had used the "SELECT * FROM "CREDITO" FULL JOIN "EMPREGO" ON "CREDITO"."Emprego" = "EMPREGO"."IDEMPREGO";" query.

