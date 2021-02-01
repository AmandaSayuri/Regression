'''
Regressao Linear Multipla
Esse código tem como objetivo utilizar o método de regressão para prever resultados de mpg, além de analisar os resultados.
A base de dados utilizada foi o dataset público auto-mpg.

Author: Amanda Sayuri Guimarães
'''


import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices


# Abertura dos dados
pd.set_option("display.max_rows", None, "display.max_columns", None)
data = pd.read_csv('data_auto-mpg.csv') 

data = data[data.horsepower != '?']
data.horsepower = data.horsepower.astype(float)
data = data.rename(columns={"model year": "year", "car name": "car_name"})

# Plot - não precisa retirar os valores que não são float
# Seleciona float automaticamente 
grr = pd.plotting.scatter_matrix(data,figsize=(15, 15), marker='o',
                                 hist_kwds={'bins': 20}, s=60, alpha=.8)


# Imprime correlações
print(data.corr())
sns.heatmap(data.corr(), annot=True)
plt.show()


# Regressao multipla
X = data[['cylinders', 'displacement', 'horsepower', 'weight',
       'acceleration', 'year', 'origin']]
Y = data[['mpg']]

X_ols = sm.add_constant(X) # adding a constant (intercept)
model = sm.OLS(Y, X_ols).fit()
predictions = model.predict(X_ols) # make the predictions by the model
model.summary()


# Alguns plots

influence = model.get_influence()
student_resid = influence.resid_studentized_external
leverage = influence.hat_matrix_diag

sns.regplot(predictions, model.resid_pearson,  fit_reg=False)
plt.title('Valor previsto vs. Studentized Residuals')
plt.xlabel('Valor previsto')
plt.ylabel('Studentized Residuals')

sns.regplot(leverage, model.resid_pearson,  fit_reg=False)
plt.title('Leverage vs. Studentized Residuals')
plt.xlabel('Leverage')
plt.ylabel('Studentized Residuals')


# Identificando multicolinearidade
data = data._get_numeric_data() 
features = "+".join(data.columns)
y, X = dmatrices('mpg~' + features, data, return_type='dataframe')
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)


# Análise final:
''' 
- Existe uma relação entre os preditores e a resposta. 
- Todas os preditores são estatiticamente significantes, com excessão do cylinders, horsepower e acceleration.
- Interessante notar que, em um aregressão simples, horsepower é significante. Mas quando interimos
os outros preditores, ele torna-se insignificante. Isso pode acontecer já que é uma variável correlacionada com 
alguns preditores
- O coeficiente do model year é 0.75. Isso significa que os modelos de carro mais modernos tem maior mpg.Para cada ano que passa, há um aumento de 0.75 unidades de mpg
- Pela análise dos resíduos, vimos que existem alguns outliers e apenas um ponto de alta alavancagem.
- Vimos também que a relação da resposta com os preditores não parece linear
'''


