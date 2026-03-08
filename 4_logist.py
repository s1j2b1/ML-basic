

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

# classification_report   مقاييس أداء
# confusion_matrix        مصفوفة الالتباس
# ConfusionMatrixDisplay  رسم مصفوفة الالتباس 
# accuracy_score          وظيفتها فقط حساب الدقة و *100 يحول النتيجة إلى نسبة مئوية      
from sklearn.metrics import classification_report , confusion_matrix , ConfusionMatrixDisplay, accuracy_score

# "True , False" ytest1لان ما يقدر يستخدم المعادلة الناقص مع "logist" ما ينفع نستخدمها مع 
from sklearn.metrics import mean_squared_error   # (y_pred - ytest)



Studied = {
    'Hours_Studied': [12, 12, 14, 15, 8, 7, 6, 2, 2, 3, 4.5, 5, 3.5, 15, 16],
    'Sleep_Hours':   [6,  5,  7,  8, 6, 6, 7, 5, 4, 6, 7, 6, 6, 8, 7],
    'Final_Grade':   [96, 95, 98, 98, 90, 89, 85, 30, 30, 50, 55, 60, 55, 98, 99]
}

data = pd.DataFrame(Studied)

print(data.head())
print(data.isnull().sum())
print(data.info())

# "True , False" نضيف عمود جديد يقسم الطلاب 
data['passed'] = data['Final_Grade'] >= 60 

x = data[['Hours_Studied','Sleep_Hours']]   # بيانات التدريب 
y = data['passed']                          # التارجت

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# او سام غير سام يعمل على التصنيف"True , False" ممكن يتعامل مع ارقام و يعطي النتيجة مثلا 
model = LogisticRegression()  # "sigmoid"  y(x) =    1
                              #                  __________
                              #                   1+ e**-x
model.fit(X_train,y_train)      
y_pred = model.predict(X_test) 
print('y_:\n',y_pred)

# حساب الدقة و يحول النتيجة إلى نسبة مئوية
print('accuracy_score:\n',accuracy_score(y_test,y_pred)*100)

'''"classification_report"
 التقرير: يحول هذه الأرقام إلى نسب مئوية سهلة الفهم مع مقاييس أداء إضافية
📊 الجدول الناتج:
support    f1-score    recall     precision     الفئة
ممنوعة        0.60     	0.75    	0.67        	4      
سليمة   	  0.80  	0.67    	0.73        	6
الدقة العامة				0.70 من أصل 10

🧠 الآن الفرق:

✅ الدقة (Precision):
كم نسبة الحقائب اللي قال عنها "ممنوعة" وكانت فعلًا ممنوعة؟
قال إن فيه 5 ممنوعة
لكن فعليًا 3 فقط كانت ممنوعة
→ الدقة = 3 / 5 = 60%

✅ الاسترجاع (Recall):
كم حقيبة ممنوعة اكتشفها النموذج من كل الممنوعات؟
فيه 4 حقائب ممنوعة فعليًا
النموذج اكتشف منها 3
→ الاسترجاع = 3 / 4 = 75%

"support" ما له علاقة إذا النموذج تنبأ صح أو خطأ، هو فقط يقول: "في الحقيقة، كم عندك من كل فئة

✅ f1-score الحسابات:
 ليش نستخدم المتوسط التوافقي <= f1
f1 = 2 × (0.60×0.75)/(0.60+0.75) ≈ 0.67                       

Harmonic Mean f1= axbx2
              ـــــــ
                a+b
أنت مدير أمن عندك جهاز فحص يكتشف "حقائب فيها متفجرات".
لو الجهاز دقته 100% (كل اللي قال إنها متفجرة طلعت فعلًا)، بس استرجاعه 10% 
(ما قدر يكتشف إلا 1 من كل 10 خطيرين). هل ترضى عنه؟ أكيد لا!
لكن لو حسبت:
المتوسط العادي: (1.0 + 0.1)/2 = 0.55

F1 =  0.1x1x2    = 0.18
      ــــــــ
       0.1+1

👮‍♂️ الوضع: في المطار فيه 50 حقيبة فيها قنابل (حقيقةً).
الشرطي استخدم الجهاز، وقال إن 5 حقائب فيها قنابل.
ولما تحققنا، طلع فعلاً كل الخمس حقائب اللي اكتشفها صحيحة (يعني كلها قنابل حقيقية).
5 / 5 = 1.0 (100%)
5 / 50 = 0.1 (فقط 10%)
المتوسط العادي = (1.0+0.1)/2=0.55
F1 =  0.1x1x2    = 0.18
      ــــــــ
       0.1+1
'''
report = classification_report(y_test,y_pred)  # "logist" طريقة احتساب الخطا في 
print('report:\n',report)

'''"confusion_matrix"
 مصفوفة الالتباس: تعطيك الأرقام الخام فقط (مثل: 8 تصنيفات صحيحة، 2 خطأ).
هي نفس الارقام الي نطلعها بعدين في الرسم 
مثال لقرائة الخلايا 
# توقع: زنبق        توقع: عباد شمش      توقع: ورد
#       حقيقة: زنبق    13                 1                    1
#       حقيقة عباد شمس     0                10                    5
#       حقيقة: ورد     0                 2                    9
'''
cm = confusion_matrix(y_test,y_pred)
print('cm:\n',cm)

# "ConfusionMatrixDisplay" جدول يوضح أداء النموذج برسم واضح بالألوان
ConfusionMatrixDisplay(cm, display_labels=model.classes_).plot(cmap='Blues')
plt.title('ConfusionMatr')
plt.show()




