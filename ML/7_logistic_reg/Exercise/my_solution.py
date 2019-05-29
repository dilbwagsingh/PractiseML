import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
from sklearn.linear_model import LogisticRegression as lr;
from sklearn.model_selection import train_test_split as tts;
from sklearn.preprocessing import  LabelEncoder;


# loading Data from file
df = pd.read_csv("HR_comma_sep.csv");
# print(df.head());

# checking if any of the entry in the file is NaN
print(df.isnull().any().any()); # this returns a boolean - if any of the data is NaN this will return True else False


# Plotting bar graph to see the relation between the salary and employee retension
low = 0;
medium =0; 
high= 0;
for i in range(len(df["left"])):
	if (df["salary"][i] == "low"):
		low += df["left"][i];
	elif (df["salary"][i] == "medium"):
		medium += df["left"][i];
	else:
		high += df["left"][i];
# print(low,medium,high);

X = ["low","medium","high"];
y = [low,medium,high];

plt.bar(X,y);
plt.xlabel("Salary category" , color = "b");
plt.ylabel("Number of employees", color = "b");
plt.show(block = False);
plt.pause(3);
plt.close();

# Plotting bar graph between department and retension
depts = dict.fromkeys(pd.unique(df["Department"]),0);
# print(depts);
for i in range(len(df["Department"])):
	depts[df["Department"][i]] += df["left"][i];
# print (depts);

plt.figure(figsize=(15, 5));
plt.ylabel("Number of employees", color ="b");
plt.xlabel("Departments", color = "b");
plt.bar(depts.keys(),depts.values());
plt.show(block = False);
plt.pause(3);
plt.close();


# Data Munging
df = df.drop(columns = ["last_evaluation", "Work_accident", "average_montly_hours"]);
le = LabelEncoder();
df.salary = le.fit_transform(df.salary);
dummies = pd.get_dummies(df.Department);
df = pd.concat([df,dummies], axis = 1);
df = df.drop(columns =["Department","RandD"]);


# Make and Train logistic regression model
X = df.drop(columns = ["left"]);
y = df.left;

# make train and test stes of data
X_train,X_test,y_train,y_test = tts(X,y,test_size=0.05);

clf = lr();
clf.fit(X_train,y_train);

pred = clf.predict(X_test);
print (pred);

acc = clf.score(X_test,y_test);
print (acc);
