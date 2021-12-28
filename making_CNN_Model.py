import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split

Fight_data = np.load('Images/Fight/Fight_0.npy')
Rob_data = np.load('Images/Rob/Rob_0.npy')

# ==================== Input ====================
Fight_data_reshape = Fight_data.reshape(-1, 400 * 300) # Assumption : Image's size 400 * 300
Rob_data_reshape = Rob_data.reshape(-1, 400 * 300) # Assumption : Image's size 400 * 300

train_input = np.append(Fight_data_reshape, Rob_data_reshape, axis=0)
print(train_input.shape)
# ==================== Input ====================

# ==================== Target ====================
Fight_target = np.full(Fight_data_reshape.shape[0], 0) # 0 : Fight
Rob_target = np.full(Rob_data_reshape.shape[0], 1) # 1 : Rob

train_target = np.append(Fight_target, Rob_target, axis=0)
print(train_target.shape)
# ==================== Target ====================

# Preprocessing
train_scaled = train_input / 255.0

train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2)

dense = keras.layers.Dense(10, activation='sigmoid', input_shape=(120000, ))
model = keras.Sequential(dense)
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(train_scaled, train_target, epochs=10)
print(model.evaluate(val_scaled, val_target))