import pandas as pd;
from sklearn import linear_model as lm;
from word2number import w2n;

df = pd.read_csv("hiring.csv");
# print(df);

# data munging
df.rename(columns = {"test_score(out of 10)" : "tscore", "interview_score(out of 10)" : "iscore", "salary($)" : "salary"} , inplace = True)
df.experience.fillna("zero", inplace = True);
median = df["tscore"].median();
df["tscore"].fillna(median, inplace= True);

# Alternatively you can do this too
# modified = [];
# for i in range(len(df["experience"])):
# 	modified.append(w2n.word_to_num(df.experience[i]));
# df.experience = modified;

df.experience = df.experience.apply(w2n.word_to_num);

# Training liner regression with multiple variables model
reg = lm.LinearRegression();
reg.fit(df[["experience","tscore","iscore"]],df["salary"]);

# Predicting salary now
a,b = reg.predict([[2,9,6],[12,10,10]]);
a = round(a);
b = round(b);
print(a,b)
