

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split     # لتقسيم الداتا
from sklearn.metrics import mean_squared_error           # اختبار فرق الصح و الخطأ

Studied = {
    'Hours_Studied': [12, 12, 14, 15, 8, 7, 6, 2, 2, 3, 4.5, 5, 3.5, 15, 16],
    'Sleep_Hours':   [6,  5,  7,  8, 6, 6, 7, 5, 4, 6, 7, 6, 6, 8, 7],
    'Final_Grade':   [96, 95, 98, 98, 90, 89, 85, 30, 30, 50, 55, 60, 55, 98, 99]
}

data = pd.DataFrame(Studied)


print(data.head())
print(data.isnull().sum())
print(data.iloc[0])
print(data.info())

x = data[['Hours_Studied','Sleep_Hours']]   # بيانات التدريب 
y = data['Final_Grade']                     # التارجت
''' x => بيانات التدريب
y => التارجت
test_size=0.2 => خذ 20% للاختبار
random_state=42 => طريقة ترتيب اخذ البيانات
'''
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# -- Train/Test Split --
'''
"XTrain" بيانات التحليل في التدريب
"XTest"  بيانات التحليل في الاختبار  
"yTrain" بيانات النتيجة المطلوبة في التدريب 
"yTest"  بيانات النتيجة المطلوبة في الاختبار

random_state=42 يعطيك نفس عملية التقسيم بالضبط كل مرة.
التقسيم يعني:
من الـ 100 عينة → 70 تروح للتدريب و 30 تروح للاختبار.
هو اللي يتحكم من يروح في أي جهة"random_state" و الـ

test_size=0.3, random_state=10 :مثلا 
teachers = ['أحمد', 'سارة', 'سالم', 'فاطمة', 'علي', 'نورة', 'يوسف', 'مها', 'راشد', 'ليلى']
print("📘 التدريب:", train)
print("🧪 الاختبار:", test)
:النتيجة
📘 التدريب: ['راشد', 'أحمد', 'علي', 'فاطمة', 'ليلى', 'سارة', 'نورة']
🧪 الاختبار: ['مها', 'يوسف', 'سالم']

في كل يختار نفس الموقع للبيانات دام انها نفس العدد حتى لو غيرت البيانات 
فهو فعليًا تقدر تعرف من راح ينختار بالضبط
'''
model = LinearRegression()    # y=b0+b1*x+.. استخدم المعادلة

model.fit(X_train,y_train)      # x النتيجة المطلوبة , بيانات التدريب y
y_pred = model.predict(X_test) # "Xtest" اعمل اختبار التوقع لقيم 

# erorr = (y_pred1 - ytest1)**2 + (y_pred2 - ytest2)**2..
mse = mean_squared_error(y_test, y_pred)  # احتساب اخطاء التوقع
print('poly_mse:',mse)
print('np.sqrt:',np.sqrt(mse)) # √erorr جذر

""" 💡 لماذا هذه المخرجات مهمة؟
تحديد أهم الخصائص: المعامل الأكبر له تأثير أقوى
فهم العلاقة:
معامل موجب → علاقة طردية
معامل سالب → علاقة عكسية
تفسير النموذج: تساعد في شرح كيف يتخذ القرارات
احيانا ممكن نحذف الخصائص ذات المعاملات القريبة من الصفر

يعطي نموذج أسرع وأبسط بنفس الأداء تقريبًا
'model.intercept_ ' b0 القيمة الأساسية 

علاقة الاعمدة بالمخرجات التأثير على المخرجات
"""
# y = b0+b1x+b2x
for feature, coef in zip(x.columns, model.coef_):
    print(f"{feature}: {coef:.2f}")     # نسبة العلاقة بين الفيتشر و المخرجات
print(f"Intercept: {model.intercept_:.2f}")


# تحدد نوع الخط،اللون،الحجم "=line_kws"
# يخلي الخط منحني يستخدم في بولونوميل ما هنا"order=2"
# x,y اكتب على المحاور \ data البيانات \ ci=None يراوينا مدى البيانات اذا شلتها 
sns.regplot(x='Hours_Studied', y='Final_Grade', data=data, order=2 , ci=None, line_kws={"color":"green"} )
plt.title("Polynomial Fit (Sleep Hours vs Grade)")
plt.show()


# شوف ليش عملنا هذا هل اله حاجة
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Data points
ax.scatter(data['Hours_Studied'], data['Sleep_Hours'], data['Final_Grade'], color='blue')

# Regression surface
x_surf, y_surf = np.meshgrid(
    np.linspace(data['Hours_Studied'].min(), data['Hours_Studied'].max(), 20),
    np.linspace(data['Sleep_Hours'].min(), data['Sleep_Hours'].max(), 20)
)
z_surf = model.intercept_ + model.coef_[0] * x_surf + model.coef_[1] * y_surf
ax.plot_surface(x_surf, y_surf, z_surf, alpha=0.5, color='red')

# Labels
ax.set_xlabel('Hours Studied')
ax.set_ylabel('Sleep Hours')
ax.set_zlabel('Final Grade')
plt.title("3D Regression Surface")
plt.show()





