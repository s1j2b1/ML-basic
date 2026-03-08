

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix, ConfusionMatrixDisplay
from ucimlrepo import fetch_ucirepo

columns = [
    'poisonous','cap_shape','cap_surface','cap_color','bruises','odor','gill_attachment',
    'gill_spacing','gill_size','gill_color','stalk_shape','stalk_root','stalk_surface_above_ring',
    'stalk_surface_below_ring','stalk_color_above_ring','stalk_color_below_ring','veil_type',
    'veil_color','ring_number','ring_type','spore_print_color','population','habitat'
]
x = pd.read_csv(r'C:\Users\Lenovo\Downloads\mushroom\agaricus-lepiota.data')
df = pd.read_csv(x, header=None, names=columns)

mushroom = fetch_ucirepo(id=73)
print(mushroom.metadata)

x = mushroom.data.features
y = mushroom.data.targets

# x = x[~x.isin(['?']).any(axis=1)]
x = x.drop('stalk_root',axis=1)
y = y['poisonous'].map({'e':0 , 'p':1})
print(y.head())
pre = columns([
    ('cat', One(handle_unknown='ignore'), x.columns)
])
model= make_pipeline(preprocessor, LinearRegression(max_iter=1000))


# print(20*'-')

# بيانات حقيقية (y_test) وتوقعات النموذج (y_pred)
y_test = [1,0,1,1,0,1]  # 1=خبيثة، 0=حميدة
y_pred = [1,0,0,1,0,1]   # توقعات النموذج

# 1. مصفوفة الارتباك
print("مصفوفة الارتباك:")
print(confusion_matrix(y_test, y_pred))

# 2. تقرير التصنيف
print("\nتقرير التصنيف:")
print(classification_report(y_test, y_pred))

# 3. عرض بياني
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("أداء النموذج في تشخيص الأورام")
plt.show()
