import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
from sklearn.linear_model import LogisticRegression as lr;
from sklearn.model_selection import train_test_split as tts;

# loading Data from file
df = pd.read_csv("HR_comma_sep.csv");
print(df.head());

# checking if any of the entry in the file is NaN
print(df.isnull().any().any()); # this returns a boolean - if any of the data is NaN this will return True else False

# Data Munging


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
plt.bar(depts.keys(),depts.values());
plt.show(block = False);
plt.pause(3);
plt.close();


