
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures



# Extended dataset with new feature: Sleep_Hours
Studied = {
    'Hours_Studied': [12, 12, 14, 15, 8, 7, 6, 2, 2, 3, 4.5, 5, 3.5, 15, 16],
    'Sleep_Hours':   [6,  5,  7,  8, 6, 6, 7, 5, 4, 6, 7, 6, 6, 8, 7],
    'Final_Grade':   [96, 95, 98, 98, 90, 89, 85, 30, 30, 50, 55, 60, 55, 98, 99]
}

data = pd.DataFrame(Studied)

# data = pd.read_csv('Student.csv')

print(data.head())
print(data.isnull().sum())
print(data.iloc[0])

# Define X (features) and y (target)
x = data[['Hours_Studied','Sleep_Hours']]   # بيانات التدريب 
y = data['Final_Grade']                     # التارجت

# Split the data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# ============ Linear Regression ============
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("🔹 Linear Regression:")
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("\nModel Coefficients:")
for feature, coef in zip(x.columns, model.coef_):
    print(f"{feature}: {coef:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

# ========= Polynomial Regression ===========
# تحويل البيانات إلى درجة 2 (x, x^2, xy, y^2, إلخ)
poly = PolynomialFeatures(degree=2)   # .. استخدم المعادلة لتحويل البيانات بالمعادلة

X_poly_train = poly.fit_transform(X_train)  # poly تحويل بيانات التدريب لتتناسب مع 
X_poly_test = poly.transform(X_test)        # poly تحويل بيانات الاختبار لتتناسب مع

# تدريب النموذج على البيانات المحولة
poly_model = LinearRegression()             # نرجع نستخدم لينير بعد ما حولنا البيانات
poly_model.fit(X_poly_train,y_train)

# التنبؤ
y_poly_pred = poly_model.predict(X_poly_test)

# تقييم الأداء
poly_mse = mean_squared_error(y_test,y_poly_pred) # احتساب الخطأ
print('poly_mse:',poly_mse)

poly_rmse = np.sqrt(poly_mse)                     # √erorr جذر
print("\n🔸 Polynomial Regression (degree=2):")
print("Mean Squared Error:", poly_mse)
print("Root Mean Squared Error:", poly_rmse)

# ========== رسم بياني توضيحي (اختياري) ==========
# "poly"لتعمل خط منحني للـ "order=2" نضيف 
sns.regplot(x='Hours_Studied', y='Final_Grade', data=data, order=2 , ci=None, line_kws={"color":"green"} )
plt.title("Polynomial Fit (Sleep Hours vs Grade)")
plt.show()


# في المثال المنزل الكبير مع غرف كثيرة قد يكون سعره أعلى من 
# مجموع سعري المنزل الكبير بغرف قليلة + المنزل الصغير بغرف كثيرة.
# كيف يقدر يعرف ان اذا المساحة زادت و زاد عدد الغرف بيزيد
#  السعر لاكن اذا المساحة نفسها و زاد عدد الغرف 
# ينقص السعر

# ما فهمت ايش تقصد بالنموذج يتذكر البيانات بدل تعلم الأنماط العامة

# ما فهمت ايش معنى  او ايش تقصد بالحدود الخطية

# طيب ايش قصة التربيع ليش نعمل تربيع ايش الي يصير 







