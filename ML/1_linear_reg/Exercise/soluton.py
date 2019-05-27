import pandas as pd;
import matplotlib.pyplot as plt;
from sklearn import linear_model as lm;
import numpy as np;


# reading data from the file
df = pd.read_csv("E:\\OpenSourceProj\\PractiseML\\ML\\1_linear_reg\\Exercise\\canada_per_capita_income.csv");
# print(df.head());


# Visualising the data
plt.scatter(df["year"],df["per capita income (US$)"],marker = "*", color = "g");
plt.show(block = False);
plt.pause(2);
plt.close();

print("\n");
print("\n");
# Training our linear regression Model
reg = lm.LinearRegression();
reg.fit(df[["year"]],df["per capita income (US$)"]);


print("\n");
print("\n");
# intercept ans slope of the trained model
intercept = reg.intercept_;
slope = reg.coef_;
print("intercept ans slope of the trained model: ",intercept, slope);


print("\n");
print("\n");
# plot the best fit line found using linear regression
plt.scatter(df["year"],df["per capita income (US$)"],marker = "*", color = "g");
plt.plot(df[["year"]], reg.predict(df[["year"]]));
plt.show(block = False);
plt.pause(2);
plt.close();


print("\n");
print("\n");
# Prediction of the per capita income in the year 2020
print("Prediction of the per capita income in the year 2020",reg.predict([[2020]]));