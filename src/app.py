# imports:
import joblib
import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


url = 'https://raw.githubusercontent.com/4GeeksAcademy/random-forest-project-tutorial/main/titanic_train.csv'
df_raw = pd.read_csv(url)

df_raw.to_csv('../data/raw//titanic_raw.csv')
df_interim = df_raw.copy()


print('El dataset está desbalanceado respecto a la varianle objetivo: Survived')
print('La métrica a mirar es entonces el f1-score preferentemente ante el accuracy')

# replace 'Age' NaNs for mean
df_interim['Age'][np.isnan(df_interim['Age'])] = df_interim['Age'].mean()

# drop 'Name', 'Ticket', 'Cabin', son todos diferents y no aportan nada:
df_interim = df_interim.drop(columns=['Name', 'Ticket', 'Cabin'])

# Transform to numerical:
# Transformo categorica 'Sex' a numérica con el dict: {'male':1, 'female':0}
df_interim['Sex'] = df_interim['Sex'].map({'male':1, 'female':0})

# Transformo categorica 'Embarked' a numérica con el dict: {'S':2, 'C':1, 'Q':0}
df_interim['Embarked'] = df_interim['Embarked'].map({'S':2, 'C':1, 'Q':0})

# Replace 'Embarked' NaNs for mode:
df_interim['Embarked'][np.isnan(df_interim['Embarked'])] = 2


df_interim.to_csv('../data/interim//titanic_raw.csv')
df = df_interim.copy()


# test-train split:
X = df.drop(columns=['Survived'])
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y , random_state=13, test_size=0.2)

'''
model_RF = RandomForestClassifier(n_estimators=50, random_state=13)
model_RF.fit(X_train, y_train)

y_train_pred = model_RF.predict(X_train)
y_test_pred = model_RF.predict(X_test)

# CLASSIFICATION REPORTs:
print(f'CLASSIFICATION REPORT on y_train_pred vs y_train: \n {metrics.classification_report(y_pred=y_train_pred, y_true=y_train)}')
print(f'CLASSIFICATION REPORT on y_train_pred vs y_test: \n {metrics.classification_report(y_pred=y_test_pred, y_true=y_test)}')
'''
# f1-score es muy bueno en train (1) y no muy bueno en test (0.75), el algoritmo está "memorizando los datos" (overfitting). 
# Entonces pruebo cambiando parámetros.

model_RT_opt = RandomForestClassifier(n_estimators=10, random_state=13, max_depth=5)
model_RT_opt.fit(X_train, y_train)

y_train_pred = model_RT_opt.predict(X_train)
y_test_pred = model_RT_opt.predict(X_test)

# CLASSIFICATION REPORTs:
print(f'CLASSIFICATION REPORT on y_train_pred vs y_train: \n {metrics.classification_report(y_pred=y_train_pred, y_true=y_train)}')
print(f'CLASSIFICATION REPORT on y_train_pred vs y_test: \n {metrics.classification_report(y_pred=y_test_pred, y_true=y_test)}')


# Export model
joblib.dump(model_RT_opt, '../models/RT_Titanic.pkl')