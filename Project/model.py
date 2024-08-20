import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

data=pd.read_csv('train.csv')
print(data.head())
from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
data['Name']=label_encoder.fit_transform(data['Name'])
data['Occupation']=label_encoder.fit_transform(data['Occupation'])
data['Credit_Mix']=label_encoder.fit_transform(data['Credit_Mix'])
data['Payment_of_Min_Amount']=label_encoder.fit_transform(data['Payment_of_Min_Amount'])
data['Payment_Behaviour']=label_encoder.fit_transform(data['Payment_Behaviour'])
data['Type_of_Loan']=label_encoder.fit_transform(data['Type_of_Loan'])
print(data.head())
x=data.drop(['Credit_Score','ID','Name','Month','Age','SSN','Occupation','Changed_Credit_Limit','Num_Credit_Inquiries','Credit_Utilization_Ratio','Delay_from_due_date','Customer_ID','Type_of_Loan'],axis=1)
y=data[['Credit_Score']]
x.info()
print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
x_train.info()
x_test.info()
from sklearn.ensemble import RandomForestClassifier
rf_cls=RandomForestClassifier()
rf_cls=rf_cls.fit(x_train,y_train)
y_pred_rf=rf_cls.predict(x_test)
print(y_pred_rf)
pickle.dump(rf_cls,open('model.pkl','wb'))