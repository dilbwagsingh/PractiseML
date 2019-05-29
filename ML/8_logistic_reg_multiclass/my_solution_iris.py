import matplotlib.pyplot as plt;
from sklearn.model_selection import train_test_split as tts;
from sklearn.datasets import load_iris;
from sklearn.linear_model import LogisticRegression as lr;
from sklearn.metrics import confusion_matrix;
import seaborn as sb;

iris = load_iris();
print(dir(iris));
# print(iris.data[0]);

X_train , X_test , y_train, y_test = tts(iris.data , iris.target, test_size = 0.1);

clf = lr();
clf.fit(X_train,y_train);

# print(y_test);
pred =  clf.predict(X_test);
# print(pred);

# acc = clf.score(X_test,y_test);
# print (acc);


cm = confusion_matrix(y_test,pred);
print(cm);

plt.figure(figsize = (10,7));
sb.heatmap(cm, annot = True);
plt.xlabel("predicted");
plt.ylabel("Truth");

plt.show();
