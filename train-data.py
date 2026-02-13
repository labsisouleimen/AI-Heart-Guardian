import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Flatten, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# 📌 تحميل البيانات
normal_data = np.loadtxt('normali_ecg_segments.csv', delimiter=',')
abnormal_data = np.loadtxt('abnormali_ecg_segments.csv', delimiter=',')

# 📌 إنشاء التسميات (Labels)
normal_labels = np.zeros((normal_data.shape[0], 1))  # 0 = طبيعي
abnormal_labels = np.ones((abnormal_data.shape[0], 1))  # 1 = غير طبيعي

# 📌 دمج البيانات والتسميات
X = np.vstack((normal_data, abnormal_data))  # الإشارات
y = np.vstack((normal_labels, abnormal_labels))  # التصنيفات

# 📌 تقسيم البيانات إلى تدريب (80%) واختبار (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 تطبيع البيانات باستخدام StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 🔹 حفظ StandardScaler لاستخدامه أثناء التنبؤ
joblib.dump(scaler, "scaler.pkl")

# 📌 إعادة تشكيل البيانات لتناسب Conv1D + LSTM (يجب أن تكون ثلاثية الأبعاد)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# 📌 تحويل y إلى One-Hot Encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)

# 📌 بناء نموذج Conv1D + LSTM
model = Sequential([
    Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(1800, 1)),
    MaxPooling1D(pool_size=2),

    Conv1D(filters=128, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),

    LSTM(64, return_sequences=True),
    LSTM(32),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')  # مخرجات التصنيف (0 = طبيعي، 1 = غير طبيعي)
])

# 📌 تجميع النموذج
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 📌 طباعة ملخص النموذج
model.summary()
# 📌 تدريب النموذج
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# 📌 تقييم النموذج على بيانات الاختبار
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'🔹 دقة الاختبار: {test_accuracy * 100:.2f}%')

# 🔹 حفظ النموذج لاستخدامه لاحقًا
model.save("ecg_cnn_lstm_model2.h5")
