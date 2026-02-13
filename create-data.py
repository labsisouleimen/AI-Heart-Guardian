import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# تحميل البيانات من ملفات CSV
normal_data = np.loadtxt('normali_ecg_segments.csv', delimiter=',')
abnormal_data = np.loadtxt('abnormali_ecg_segments.csv', delimiter=',')

# إنشاء التسميات (Labels)
normal_labels = np.zeros((normal_data.shape[0], 1))  # 0 = طبيعي
abnormal_labels = np.ones((abnormal_data.shape[0], 1))  # 1 = غير طبيعي

# دمج البيانات والتسميات معًا
X = np.vstack((normal_data, abnormal_data))  # دمج الإشارات الطبيعية وغير الطبيعية
y = np.vstack((normal_labels, abnormal_labels))  # دمج التسميات

# تقسيم البيانات إلى تدريب (80%) واختبار (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تطبيع البيانات باستخدام StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# تحويل y إلى تصنيف فئتين (One-Hot Encoding)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)
print("all is good")
