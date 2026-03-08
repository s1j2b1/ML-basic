
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # اضف الرسم
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, cross_val_score

# cross_val_score تعمل تقسم و تدريب و اختبار و تعطي الدقة
from sklearn.preprocessing import MinMaxScaler   # تحويل البيانات من 1-100 الى 0-1
from sklearn.datasets import load_breast_cancer  # "datasets" استدعاء داتا من مكتبة 



data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
print(df.head())

# (X min - X) ÷ (X max - X min) "معادلة تحويل القيم الى قيم بين "0الى1
scaler = MinMaxScaler()  
x_scaled = scaler.fit_transform(df.drop('target', axis=1))
y = df['target']

model = LogisticRegression()
# الطريقة1
# X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)
# model.fit(X_train,y_train)
# y_pred = model.predict(X_test)
# print('report:\n',classification_report(y_test,y_pred))  # "logist" طريقة احتساب الخطا في

# الطريقة2
# (عدد التقسيمات , التارجت , بيانات التدريب , الخوارزمية)
# تعمل تقسم و تدريب و اختبار و تعطي الدقة اما النسبة80٪ 20٪ تعتمد على كم عدد الجولات
scores = cross_val_score(model, x_scaled,y , cv=5) # تقسيم بيانات التدريب الى 5 مراحل
print('score:\n',scores)  # الدقة في كل جولة
print(scores,'mean:', np.mean(scores))



