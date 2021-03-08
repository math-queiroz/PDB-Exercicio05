# Main Dependancies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Sklearn Dependancies
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Datasaet Import
dataset = pd.read_csv('dados.csv')
ind = dataset.iloc[:, 0].values
dep = dataset.iloc[:, -1].values

# Reshape
ind = ind.reshape(len(ind), 1)

# LinearRegression Model Instance and Fit
linearRegression = LinearRegression()
linearRegression.fit(ind, dep)

# PolynomialFeatures instance and Fit
poly_features = PolynomialFeatures(degree=2)
ind_poly = poly_features.fit_transform(ind)
polyLinearRegression = LinearRegression()
polyLinearRegression.fit(ind_poly, dep)

# Gráfico do Modelo Linear Simples
plt.scatter(ind, dep, color='red')
plt.plot(ind, linearRegression.predict(ind), color='blue')
plt.title('Regressão Linear Simples')
plt.xlabel('Idade')
plt.ylabel('Altura')
plt.show()

# Gráfico do Modelo Linear Polinomial
plt.scatter(ind, dep, color='red')
plt.plot(ind, polyLinearRegression.predict(ind_poly), color='blue')
plt.title('Regressão Linear Polinomial (degree=2)')
plt.xlabel('Idade')
plt.ylabel('Altura')
plt.show()

# Ajustando a Regressão Polinomial para grau 4
poly_features.degree = 4
ind_poly = poly_features.fit_transform(ind)
polyLinearRegression.fit(ind_poly, dep)

# Gráfico do Modelo Linear Polinomial
plt.scatter(ind, dep, color='red')
plt.plot(ind, polyLinearRegression.predict(ind_poly), color='blue')
plt.title('Regressão Linear Polinomial (degree=4)')
plt.xlabel('Idade')
plt.ylabel('Altura')
plt.show()
