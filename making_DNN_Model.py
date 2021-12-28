import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
import pandas as pd

CSV_Assault = pd.read_csv("CSV/Raw/Assault.csv")
CSV_Broke = pd.read_csv("CSV/Raw/Broke.csv")
CSV_Burglary = pd.read_csv("CSV/Raw/Burglary.csv")
CSV_Normal = pd.read_csv("CSV/Raw/Normal.csv")

# CSV_Assault = pd.read_csv("CSV/Scaled/Assault_revised.csv")
# CSV_Broke = pd.read_csv("CSV/Scaled/Broke_revised.csv")
# CSV_Burglary = pd.read_csv("CSV/Scaled/Burglary_revised.csv")
# CSV_Normal = pd.read_csv("CSV/Scaled/Normal_revised.csv")

# Input data
Assault_input = CSV_Assault[["RElbow_x", "RElbow_y", "RElbow_z",
                             "RShoulder_x", "RShoulder_y", "RShoulder_z",
                             "RWrist_x", "RWrist_y", "RWrist_z",
                             "RHip_x", "RHip_y", "RHip_z",
                             "RKnee_x", "RKnee_y", "RKnee_z",
                             "RAnkle_x", "RAnkle_y", "RAnkle_z",
                             "LElbow_x", "LElbow_y", "LElbow_z",
                             "LShoulder_x", "LShoulder_y", "LShoulder_z",
                             "LWrist_x", "LWrist_y", "LWrist_z",
                             "LHip_x", "LHip_y", "LHip_z",
                             "LKnee_x", "LKnee_y", "LKnee_z",
                             "LAnkle_x", "LAnkle_y", "LAnkle_z"
                             ]].to_numpy()
Broke_input = CSV_Broke[["RElbow_x", "RElbow_y", "RElbow_z",
                         "RShoulder_x", "RShoulder_y", "RShoulder_z",
                         "RWrist_x", "RWrist_y", "RWrist_z",
                         "RHip_x", "RHip_y", "RHip_z",
                         "RKnee_x", "RKnee_y", "RKnee_z",
                         "RAnkle_x", "RAnkle_y", "RAnkle_z",
                         "LElbow_x", "LElbow_y", "LElbow_z",
                         "LShoulder_x", "LShoulder_y", "LShoulder_z",
                         "LWrist_x", "LWrist_y", "LWrist_z",
                         "LHip_x", "LHip_y", "LHip_z",
                         "LKnee_x", "LKnee_y", "LKnee_z",
                         "LAnkle_x", "LAnkle_y", "LAnkle_z"
                         ]].to_numpy()
Burglary_input = CSV_Burglary[["RElbow_x", "RElbow_y", "RElbow_z",
                               "RShoulder_x", "RShoulder_y", "RShoulder_z",
                               "RWrist_x", "RWrist_y", "RWrist_z",
                               "RHip_x", "RHip_y", "RHip_z",
                               "RKnee_x", "RKnee_y", "RKnee_z",
                               "RAnkle_x", "RAnkle_y", "RAnkle_z",
                               "LElbow_x", "LElbow_y", "LElbow_z",
                               "LShoulder_x", "LShoulder_y", "LShoulder_z",
                               "LWrist_x", "LWrist_y", "LWrist_z",
                               "LHip_x", "LHip_y", "LHip_z",
                               "LKnee_x", "LKnee_y", "LKnee_z",
                               "LAnkle_x", "LAnkle_y", "LAnkle_z"
                               ]].to_numpy()
Normal_input = CSV_Normal[["RElbow_x", "RElbow_y", "RElbow_z",
                           "RShoulder_x", "RShoulder_y", "RShoulder_z",
                           "RWrist_x", "RWrist_y", "RWrist_z",
                           "RHip_x", "RHip_y", "RHip_z",
                           "RKnee_x", "RKnee_y", "RKnee_z",
                           "RAnkle_x", "RAnkle_y", "RAnkle_z",
                           "LElbow_x", "LElbow_y", "LElbow_z",
                           "LShoulder_x", "LShoulder_y", "LShoulder_z",
                           "LWrist_x", "LWrist_y", "LWrist_z",
                           "LHip_x", "LHip_y", "LHip_z",
                           "LKnee_x", "LKnee_y", "LKnee_z",
                           "LAnkle_x", "LAnkle_y", "LAnkle_z"
                           ]].to_numpy()

action_input = np.append(Assault_input, Broke_input, axis=0)
action_input = np.append(action_input, Burglary_input, axis=0)
action_input = np.append(action_input, Normal_input, axis=0)

# Target data
Assault_target = CSV_Assault['Action'].to_numpy()
Broke_target = CSV_Broke['Action'].to_numpy()
Burglary_target = CSV_Burglary['Action'].to_numpy()
Normal_target = CSV_Normal['Action'].to_numpy()

action_target = np.append(Assault_target, Broke_target, axis=0)
action_target = np.append(action_target, Burglary_target, axis=0)
action_target = np.append(action_target, Normal_target, axis=0)
print(np.unique(action_target))

# 정규화
ss = StandardScaler()
ss.fit(action_input)
action_scaled = ss.transform(action_input)

# 섞기
train_input, val_input, train_target, val_target = train_test_split(action_scaled, action_target, test_size=0.2)
print(train_target[0:10])
# DNN Structure
model = keras.Sequential()
model.add(keras.layers.Dense(30, activation='relu'))
model.add(keras.layers.Dense(20, activation='relu'))
model.add(keras.layers.Dense(10, activation='relu'))
model.add(keras.layers.Dense(4, activation='softmax'))

#keras.utils.plot_model(model)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
checkpoint_cb = keras.callbacks.ModelCheckpoint('best_DNN_Raw_model.h5')
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
history = model.fit(train_input, train_target, epochs=100, validation_data=(val_input, val_target), callbacks=[checkpoint_cb, early_stopping_cb])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
# plt.legend['train', 'val']
plt.show()

