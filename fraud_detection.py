#%%
import numpy as np 
import pandas as pd 
import  matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression 


# %%
df = pd.read_csv('payment_fraud.csv')
df.head()
# %%
df.isnull().sum()
# %%
paymthd = df.paymentMethod.value_counts()
plt.figure(figsize=(5, 5))
plt.bar(paymthd.index,paymthd)
plt.ylabel('Count')
df.label.value_counts()
# %%
# paymthd_label = {v:k for k, v in enumerate(df.paymentMethod.unique())}
paymthd_label, unique_values = pd.factorize(df.paymentMethod)
df.paymentMethod =paymthd_label
print(df.head())
# %%
print(df.describe())
corr = df.corr()
corr.style.background_gradient()
# %%
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
sc = StandardScaler()
X = sc.fit_transform(x)
# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
# print("X_train shape: ", x_train.shape)
# print("X_test shape: ", x_test.shape)
# print("y_train shape: ", y_train.shape)
# print("y_test shape: ", y_test.shape)
# %%
lg = LogisticRegression()
lg.fit(x_train, y_train)

# %%
pred = lg.predict(x_test)
print("accuracy score:- " ,accuracy_score(y_test, pred))
print(classification_report(y_test, pred))


# %%
