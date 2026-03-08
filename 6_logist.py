# OneHotEncoder اعمل طبااعة للبراميتر لما ترو و لما فولس

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
# cross_val_score تعمل تقسيم و تدريب و اختبار و تعطي الدقة
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import  accuracy_score, classification_report , confusion_matrix , ConfusionMatrixDisplay
# يحول البيانات الي تكون حروف الى ارقام
from sklearn.preprocessing import OneHotEncoder

# uci.com طرق قرائة الداتا من
# كذا ما يحتاج تنزلها import in python ربطها عن طريق مكتبة تحصلها في الموقع 
# راح تنزل 2 ملفات واحد البيانات و الثاني لاسماء الاعمدة ".data" ممكن تنزلها و اذا كانت 
from ucimlrepo import fetch_ucirepo 


mushroom = fetch_ucirepo(id=73) 

# data (as pandas dataframes) 
x = mushroom.data.features 
y = mushroom.data.targets 


print('metadata:\n',mushroom.metadata)    # معلومات عن الداتا
print('variables:\n',mushroom.variables)  # معلومات تفصيلية عن كل عمود

print('info:\n',x.info())
print('x.head:\n',x.head())
print('y.head:\n',y.head())
print('columns:\n',x.columns)

# ياخذ الصفوف الي ما فيها علامة ? في اي عمود و يعكسها على الداتا
x = x[~x.isin(['?']).any(axis=1)]
y = y.loc[x.index]  # "x" و يخزن فقط الصفوف الي اسماهن نفس "targets"يرجع لـ
x = x.drop('stalk-root', axis=1)
y = y['poisonous'].map({'e':0, 'p':1})  # اشان كذا حددنا "DataFrame" ما تتعامل مع "map" هنا 

encoder = OneHotEncoder()
x = encoder.fit_transform(x)  # من حروف لارقام جاهزة للتدريب x تحويل بيانات 

model = LogisticRegression()

# الطريقة1
scores = cross_val_score(model,x,y , cv=5) # تقسيم بيانات التدريب الى 5 مراحل
print('score:\n',scores)                   # الدقة في كل جولة
print(scores,'mean:', np.mean(scores))

# الطريقة2
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model.fit(X_train,y_train)      
y_pred = model.predict(X_test) 

print('accuracy_score:\n',accuracy_score(y_test,y_pred)*100)
report = classification_report(y_test,y_pred)  # "logist" طريقة احتساب الخطا في 
print('report:\n',report)


cm = confusion_matrix(y_test,y_pred)
print('cm:\n',cm)

# "ConfusionMatrixDisplay" جدول يوضح أداء النموذج برسم واضح بالألوان
ConfusionMatrixDisplay(cm, display_labels=model.classes_).plot(cmap='Blues')
plt.title('ConfusionMatr')
plt.show()










