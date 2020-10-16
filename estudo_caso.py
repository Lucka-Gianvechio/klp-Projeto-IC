import sklearn as sk
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#CHAS E RAD são int64, o resto é float64
#sns.countplot para valores discretos, sns.histogram para contínuos
#apresentar boxplot junto com os histogramas - análise descritiva - não paramétricas

###análises descritivas
boston = datasets.load_boston(return_X_y = False)
#print(boston['feature_names'])
#print(boston.feature_names)
#print(boston.DESCR)    #descreve o dataset
#print(boston.target)     #506 linhas e 13 colunas
dataframe = pd.DataFrame(boston.data, columns=boston.feature_names)
dataframe['PRICE'] = boston.target
dataframe = dataframe.astype({'CHAS': int, 'RAD': int})
colunas_continuas = ['CRIM', 'ZN', 'INDUS','NOX', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'PRICE']

'''
fig, ax = plt.subplots(nrows = 4, ncols = 3, figsize=(20,20))
i,j = 0, 0
k = 0
####Gráfico conjunto das colunas contínuas com o preço médio das casas
for i in range(4):
    for j in range(3):
        sns.histplot(x = dataframe[colunas_continuas[k]], ax = ax[i][j])
        k += 1

plt.tight_layout()
plt.show()'''

'''
i = 0
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(10,10))
for coluna_d in colunas_discretas:
    plot = sns.boxplot(x = dataframe[coluna_d], y = dataframe['PRICE'], ax = ax[i])
    plt.show
    i += 1

plt.tight_layout()
plt.show()
'''

'''f, p_value = stats.f_oneway(dataframe['RAD'], dataframe['PRICE'])
print(f, '     ', p_value)

print(stats.shapiro(dataframe['RAD']))'''

#print(dataframe.info())
#print(dataframe.describe())
#print(dataframe.dtypes)
#print("\n\n")
#boxplot = sns.boxplot(data = dataframe)    #boxplot
#plt.show()
#plt.title("Bolxplot da idade")
#dist = sns.jointplot(dataframe['RAD'], dataframe['TAX'])

#plt.show()
#sns.countplot(x = dataframe['CHAS'])
#plt.show()
#sns.histplot(dataframe['PRICE'])
#plt.show()

features = dataframe.drop(columns = ['PRICE'])
target = dataframe['PRICE']
scalar = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.2, random_state = 600)

'''
print()
print("X_TREINO: ")
print(X_train, '\n')
print('X_TESTE: ')
print(X_test, '\n')
print('Y_TREINO: ')
print(y_train, '\n')
print('Y_TESTE: ')
print(y_test, '\n')
print()
'''
###Separação em variáveis de treino e de teste para features e target
X_train2, X_val, y_train2, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 528)

print()
print("X_TREINO: ")
print(X_train2, '\n')
print('X_TESTE: ')
print(X_val, '\n')
print('Y_TREINO: ')
print(y_train2, '\n')
print('Y_TESTE: ')
print(y_val, '\n')
print()

###Padronização
##Descartando features binárias

nao_binario_xtrain = X_train2.drop(columns = ['CHAS'])
nao_binario_xval = X_val.drop(columns = ['CHAS'])

#features não binárias padronizadas - média 0 e variância 1
x_train2 = scalar.fit_transform(nao_binario_xtrain)
x_val2 = scalar.fit_transform(nao_binario_xval)

print()
print("x_train2: ", x_train2.shape)
print("x_val2: ", x_val2.shape)
print("y_train2: ", y_train2.shape)
print("y_val2: ", y_val.shape)
print()

##set os possíveis valores para o hiperparâmetro da regressão Lasso
hiperparametros = 10 ** np.linspace(3, -10, num = 500)

###### REGULARIZAÇÃO LASSO ######

## Fazer um loop nos possíveis valores para os hiperparâmetros, sendo que a cada iteração você treinará um modelo com o conjunto (X_train2, y_train2) e avaliará
## o Erro Quadrático Médio (MSE) no conjunto de validação (X_val, y_val) - será necessáio utilizar a função 'predict' e 'mean_squared_error'.
## Armazene os melhores valores dos hiperparâmetros para a regressão Ridge e Lasso.
print("MODELO LASSO: ", '\n')

dic_mse = {}
modelo_lasso = Lasso()
modelo_ridge = Ridge()

###encontra o alpha associado ao menor MSE - Regularização Lasso
'''
for i in range(len(hiperparametros)):
    modelo_lasso.set_params(alpha = hip_lasso[i])
    modelo_lasso.fit(X_train2, y_train2)
    modelo_lasso_predito = modelo_lasso.predict(X_train2)
    mse = mean_squared_error(y_train2, modelo_lasso_predito)
    dic_mse.update({hip_lasso[i] : mse})
'''
alpha_lasso = 0.34765
alpha_ridge = 0.576
#alpha_ridge = 6.662654524581163e-09

###encontra o alpha com menor MSE - Regularização Ridge
'''
for i in range(len(hiperparametros)):
    modelo_ridge.set_params(alpha = hiperparametros[i])
    modelo_ridge.fit(X_train2, y_train2)
    modelo_ridge_predito = modelo_ridge.predict(X_train2)
    mse = mean_squared_error(y_train2, modelo_ridge_predito)
    dic_mse.update({hiperparametros[i] : mse})
'''

#RIDGE
modelo_ridge.set_params(alpha = alpha_ridge)
modelo_ridge.fit(x_train2, y_train2)
modelo_ridge_predito = modelo_ridge.predict(x_train2)
modelo_ridge_teste = modelo_ridge.predict(x_val2)
print("MSE (DADOS DE TESTE) - RIDGE:  ", mean_squared_error(modelo_ridge_predito, y_train2))

#LASSO
modelo_lasso.set_params(alpha = alpha_lasso)
modelo_lasso.fit(x_train2, y_train2)
modelo_lasso_predito = modelo_lasso.predict(x_train2)
print("MSE (DADOS DE TESTE) - LASSO:  ", mean_squared_error(modelo_lasso_predito, y_train2))

#REG LIN
reg_linear = LinearRegression().fit(x_train2, y_train2)
modelo_linear = reg_linear.predict(x_train2)
print("MSE (DADOS DE TESTE) - REG. LIN:  ", mean_squared_error(modelo_linear, y_train2))

print(" * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ")
data = {'Treino' : y_train2,'Regressão Linear' : modelo_linear,  'Teste - Lasso' : modelo_lasso_predito, 'Teste - Ridge' : modelo_ridge_predito}
data2 = {'Y - TESTE' : y_val, 'Y - PREDITO (RIDGE)' : modelo_ridge_teste}
diferenca = pd.DataFrame(data)
diferenca2 = pd.DataFrame(data2)
print(diferenca.head(20))
print(diferenca2.head(20))

#fig, ax = plt.subplots(nrows = 2, ncols = 1)

#sns.regplot(x = X_train2['RM'], y = y_train2, ax = ax[1], color = 'blue')


#sns.regplot(x = X_train2['RM'], y = modelo_lasso_predito, ax = ax[0], color = 'red')


#plt.show()

'''
classificado = sorted(dic_mse.items(), key = lambda x: x[1])
alpha, mse = classificado[0]
print(f'Alpha: {alpha}    --------  MSE: {mse}')
'''


nao_binario_xtrain = X_train.drop(columns = ['CHAS'])
nao_binario_xval = X_test.drop(columns = ['CHAS'])

#features não binárias padronizadas - média 0 e variância 1
X_train = scalar.fit_transform(nao_binario_xtrain)
X_test = scalar.fit_transform(nao_binario_xval)

print(" Tamanhos ")
print('X_train', X_train.shape)
print('X_test', X_test.shape)
print('y_train', y_train.shape)
print('y_test', y_test.shape)
print("****************************************************")
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
modelo_linear_predito = linear_model.predict(X_train)
modelo_linear_preditot = linear_model.predict(X_test)
print(linear_model.coef_)
print('modelo_linear_preditot', modelo_linear_preditot.shape)


###Regularização Lasso
X_train, X_test = scalar.fit_transform(X_train), scalar.fit_transform(X_test)
alpha_lasso = 1
lasso = Lasso(max_iter=10000)
lasso.set_params(alpha = alpha_lasso)
lasso.fit(X_train, y_train)
lasso_predito = lasso.predict(X_train)
print("LASSO COEF: ", lasso.coef_)
mse = mean_squared_error(lasso_predito, y_train)
print("MSE - DADOS TREINO : " , mse)
lasso_preditot = lasso.predict(X_test)
mse = mean_squared_error(lasso_preditot, y_test)
print("MSE - DADOS TESTE: ", mse)

###plot conjunto dos dados de teste com reg. lasso
sns.jointplot(x = y_test, y = lasso_preditot)
plt.show()
corr = np.corrcoef(y_test, lasso_preditot)
#print("correlação: ", corr)
print('lasso_predito', lasso_predito.shape)

data = {'Y Treino' : y_train , 'R.L - Y predito' : modelo_linear_predito,'Regularização Lasso' : lasso_predito}
dataframe = pd.DataFrame(data)
print(dataframe)

print("\n\n")

data = {'Y Teste' : y_test, 'R.L - Teste' : modelo_linear_preditot, 'Regularização Lasso' : lasso_preditot}
dataframe = pd.DataFrame(data)
print(dataframe)























#colunas_continuas = ['CRIM', 'ZN', 'INDUS','CHAS' ,'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'PRICE']

'''Calcular e plotar a matriz de correlação entre as features. Para a visualização foi utilizada a função heatmap() do pacote seaborn.'''
#matriz_corr = dataframe.corr().round(2)
#print('\n', matriz_corr, '\n')
#mapa_calor = sns.heatmap(matriz_corr, cmap = 'magma_r', annot=True)
#plt.title('MAPA DE CALOR DE CORRELAÇÃO')
#plt.show()
#print(matriz_corr[0])

###Devolve todas as correlações >= valor em pares
'''
lista = []
for i in range(len(matriz_corr)):
    for j in range(i, len(matriz_corr)):
        if abs(matriz_corr[colunas_continuas[i]][colunas_continuas[j]]) >= 0.5 and i != j:
            lista.append((colunas_continuas[i], colunas_continuas[j]))

print(lista)
'''
