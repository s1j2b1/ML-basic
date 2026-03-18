

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, mean_squared_error

plt.ion()  # ← لتفعيل الوضع التفاعلي لعرض الرسومات مباشرة


data = pd.read_csv(r"D:\..\pint.csv")
print(data)

data['sleep_hours'] = [6,5,7,8,6,6,7,5,4,6,7,6,6,8,7]
data['passed'] = data['Final_Grade'] >= 60
print(f"head: \n{data.head()} \n")
print(f"tail: \n{data.tail()} \n")
print(f"info: \n{data.info()} \n")

x= data[['Hours_Studied','sleep_hours']]
y= data['passed']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.2, random_state=42)

model= LogisticRegression()
model.fit(x_train,y_train)
ypredict = model.predict(x_test)

cm = confusion_matrix(y_test,ypredict)
ConfusionMatrixDisplay(cm, display_labels= model.classes_).plot(cmap='Blues')
plt.title("confusion_matrix")
plt.ion()

plt.show()
input("اضغط Enter لإغلاق الرسم...")



















