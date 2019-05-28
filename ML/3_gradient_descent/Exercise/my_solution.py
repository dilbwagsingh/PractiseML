import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
import math; 



def gradDes(X,y):
	slope = 0;
	intercept = 0;
	m = np.size(X);
	optimIterations = 0;
	alpha = 0.0001;
	prevcost = 0;
	while True:
		prediction = slope*X + intercept;
		cost = (1/(2*m))*np.sum([val**2 for val in (y-prediction)]);
		der_slope = -(1/m)*np.sum(X*(y-prediction));
		der_intercept = -(1/m)*np.sum(y-prediction);
		mod_slope = slope - alpha*der_slope;
		mod_intercept = intercept - alpha*der_intercept;
		print("Slope: ",mod_slope," Intercept ",mod_intercept," Cost: ", cost," Iteration no. : ", optimIterations);
		slope = mod_slope;
		intercept = mod_intercept;
		optimIterations += 1;
		if(math.isclose(prevcost,cost,rel_tol = 1e-9)):
			break;
		prevcost = cost;






# Importing data
df = pd.read_csv("test_scores.csv");
# print(df);

# Visualising the data
plt.scatter(df.math,df.cs, marker = "*" , color ="g");
plt.xlabel("Math scores");
plt.ylabel("CS scores");
plt.show(block = False);
plt.pause(2)
plt.close();

# Implementing Gradient descent
gradDes(df.math,df.cs);