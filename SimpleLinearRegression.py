'''
Regressao Linear Simples
Esse código tem como objetivo utilizar o método de regressão simples para preverresultados de mpg utilizando os dados de horsepower e analisar os resultados.
A base de dados utilizada foi o dataset público auto-mpg.

Author: Amanda Sayuri Guimarães
'''


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from regressors import stats
import pandas as pd
import statsmodels.api as sm
import seaborn as sns


# Captura dos dados
data = pd.read_csv('data_auto-mpg.csv') 
data = data[data.horsepower != '?']
X = data[['horsepower']].astype(float)
Y = data[['mpg']].astype(float)


# Aplicando regressao simples
regr = linear_model.LinearRegression()
regr.fit(X, Y)
Y_pred = regr.predict(X)


# Imprime resumo
X_ols = sm.add_constant(X) # adding a constant (intercept)
model = sm.OLS(Y, X_ols).fit()
predictions = model.predict(X_ols) # make the predictions by the model
model.summary()

# Prevendo um novo valor
teste = X_ols[X_ols['horsepower'] == 98]
result = model.get_prediction(teste)
result.conf_int()


# Alguns plots 
plt.title('horsepower vs mpg')
plt.scatter(X, Y,  color='black')
plt.plot(X, Y_pred, color='blue', linewidth=3)
plt.xlabel('X')
plt.ylabel('Valor Predito')
plt.show()

influence = model.get_influence()
student_resid = influence.resid_studentized_external
leverage = influence.hat_matrix_diag

sns.regplot(Y_pred, model.resid_pearson,  fit_reg=False)
plt.title('Valor previsto vs. Studentized Residuals')
plt.xlabel('Valor previsto')
plt.ylabel('Studentized Residuals')

sns.regplot(leverage, model.resid_pearson,  fit_reg=False)
plt.title('Leverage vs. Studentized Residuals')
plt.xlabel('Leverage')
plt.ylabel('Studentized Residuals')


# Análise final:
''' 
- Existe uma relação clara entre o preditor e a resposta. 
- 60% da variação do mpg é explicado por horsepower 
- Quanto maior o horsepower, menor o mgp 
- Caso horsepower seja = 98, o mpg será de 24.45
- Pelo gráfico de resíduos, vemos que a relação não é linear
- Vemos também que existem outliers e pontos de alta alavancagem
'''

