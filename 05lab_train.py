import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import sys

epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 5
batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 32

train_df = pd.read_csv("data/train.csv")
valid_df = pd.read_csv("data/val.csv")


X_train = train_df.drop(columns=["DRK_YN"])
y_train = train_df["DRK_YN"]
X_valid = valid_df.drop(columns=["DRK_YN"])
y_valid = valid_df["DRK_YN"]

model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid") 
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    epochs=epochs,
    batch_size=batch_size,
    verbose=1
)

model.save("myModel.h5")
print("Model zapisany jako myModel.h5")