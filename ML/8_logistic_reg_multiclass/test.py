import matplotlib.pyplot as plt;
from sklearn.model_selection import train_test_split as tts;
from sklearn.linear_model import LogisticRegression as lr;
from sklearn.metrics import confusion_matrix;
import seaborn as sb;
from sklearn.datasets import load_digits;

digits  = load_digits();
print (dir(digits));


# print(digits.target_names[0]);
# plt.gray();
# for i in range(5):
# 	plt.matshow(digits.images[i]);
# 	plt.gray();
# 	plt.show()

# print (digits.target[0:5]);

X_train , X_test , y_train , y_test = tts(digits.data,digits.target,test_size = 0.2);
# print (len(X_train));
clf = lr();
clf.fit(X_train,y_train);

pred = clf.predict(X_test);
# print (y_test);
# print (pred);

# acc = clf.score(X_test,y_test);
# print(acc);

# visualising my model's predictions using confusion matrix metric
cm = confusion_matrix(y_test,pred);
print (cm);

plt.figure(figsize = (10,7));
sb.heatmap(cm,annot = True);
plt.xlabel("predicted value");
plt.ylabel("True value");

plt.show();