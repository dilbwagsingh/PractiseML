import pandas as pd;
import sklearn.linear_model as lm;
import matplotlib.pyplot as plt;
from sklearn.model_selection import train_test_split


# Reading data from file
df = pd.read_csv("carprices.csv");
# print(df);

# data is already munged and is ready to be used

# Data visualisation 
plt.scatter(df["Mileage"],df["Sell Price($)"],marker = "*", color = "g");
plt.show(block = False);
plt.pause(2);
plt.close();

plt.scatter(df["Age(yrs)"],df["Sell Price($)"],marker = "o", color = "b");
plt.show(block = False);
plt.pause(2);
plt.close();

# Model training
model = lm.LinearRegression();
X = df[["Mileage","Age(yrs)"]]
y = df["Sell Price($)"];

# creating test and train datasets
X_train, X_test , y_train, y_test = train_test_split(X,y,test_size=0.2);
model.fit(X_train,y_train);

# making predictions on the test data
prediction = model.predict(X_test);
# print(y_test);
# print(prediction);

# checking accuracy of my model
acc = model.score(X_test,y_test);
print(acc);