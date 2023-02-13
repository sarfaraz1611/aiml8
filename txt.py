import numpy as np
import pandas as pd
df=pd.read_csv('Social_Network_Ads.csv')
df.head()
gen={'Male':1,'Female':0}
df.Gender=[gen[item]for item in df.Gender]
df.head()
X=df[['Age','EstimatedSalary']]
Y=df['Purchased']
print(X,"\n",Y)
X.insert(0,'B0',1)
xt=X.T
print("X",X,"\nXT",xt)
XTX=np.dot(xt,X)
print("XTX",XTX)
XTXI=np.linalg.inv(XTX)
print("XTXI",XTXI)
XTY=np.dot(xt,Y)
print("XTY",XTY)
BHAT=np.dot(XTXI,XTY)
print("BHAT",BHAT)
age = 36
salary = 76000
y_pred=BHAT[0]+BHAT[1]*age+BHAT[2]*salary
print("Y_PRED",y_pred)
prediction=1/(1+(2.718)**-y_pred)
print("Value of sigmoid function : ",prediction)
if prediction>0.5:
 print("PREDICTION : YES")
else:
 print("PREDICTION : NO")
