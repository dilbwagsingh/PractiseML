import pandas as pd;
from sklearn import linear_model as lm;


# Reading data
df = pd.read_csv("carprices.csv");
# print(df);

# Data Munging
dummies = pd.get_dummies(df["Car Model"]);
# print(dummies);

merged = pd.concat([df,dummies],axis = "columns"); 
# or you can instead write df = pd.concat([df,dummies],axis = 1);
# this behaves as if in (rows,columns) columns = 1
# print(merged.head());

# final is the munged data
# we remove "Mercedez Benz C class" due to a problem called dummy variable trap, Read more form Google
final = merged.drop(["Car Model","Mercedez Benz C class"],axis =1);
# print(final);

X = final.drop(["Sell Price($)"], axis = 1);
# print(X);
y = final["Sell Price($)"];
# print (y);

# Model training
model = lm.LinearRegression();
model.fit(X,y);

# Using model to predict selling prices
prediction = model.predict([[45000,4,0,0],[86000,7,0,1]]);
print(prediction);

# Accuracy of my trained model in percent
acc = model.score(X,y) * 100;
print(acc,"%");