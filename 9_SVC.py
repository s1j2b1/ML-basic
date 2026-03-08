# SVC مو مستخدمة او مو مستخدممة كثير بس تنسئل في المقابلات ف افهمها 

import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report , confusion_matrix , ConfusionMatrixDisplay , accuracy_score
from sklearn.svm import SVC


mushroom = fetch_ucirepo(id=73) 

# data (as pandas dataframes) 
x = mushroom.data.features 
y = mushroom.data.targets 


print('info:\n',x.info())
print('x.head:\n',x.head())
print('y.head:\n',y.head())
print('columns:\n',x.columns)

x = x[~x.isin(['?']).any(axis=1)]
y = y.loc[x.index]  
x = x.drop('stalk-root', axis=1)
y = y['poisonous'].map({'e':0, 'p':1}) 

encoder = OneHotEncoder(handle_unknown='ignore')
x = encoder.fit_transform(x)

model = SVC(kernel='linear')

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model.fit(X_train,y_train)
y_pred = model.predict(X_test) 


print(accuracy_score(y_test,y_pred)*100)

report = classification_report(y_test,y_pred)  
print('report:\n',report)

cm = confusion_matrix(y_test,y_pred)
print('cm:\n',cm)

ConfusionMatrixDisplay(cm, display_labels=model.classes_).plot(cmap='Blues')
print('model.classes_:\n',model.classes_)
plt.title('ConfusionMatr')
plt.show()




