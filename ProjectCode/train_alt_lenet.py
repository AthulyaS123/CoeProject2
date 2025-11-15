import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np

# ---------- 1. Load dataset from directory ----------
dataset = tf.keras.utils.image_dataset_from_directory(
    "Project2",                 # folder with damage/ and no_damage/
    labels="inferred",
    label_mode="int",
    image_size=(128, 128),
    color_mode="rgb",
    batch_size=64,
    shuffle=True,
    seed=1,
)

class_names = dataset.class_names
print("Class names:", class_names)

# Convert tf.data.Dataset -> full NumPy arrays
X_list = []
y_list = []
for batch_images, batch_labels in dataset:
    X_list.append(batch_images.numpy())
    y_list.append(batch_labels.numpy())

X = np.concatenate(X_list, axis=0)
y = np.concatenate(y_list, axis=0)

print("X shape:", X.shape)
print("y shape:", y.shape)

# ---------- 2. Normalize and split ----------
X = X / 255.0

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=1
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    stratify=y_temp,
    random_state=1
)

print("Train:", X_train.shape, y_train.shape)
print("Val:  ", X_val.shape,   y_val.shape)
print("Test: ", X_test.shape,  y_test.shape)

# ---------- 3. Define Alternate-LeNet-5 architecture ----------
def build_alt_lenet(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", padding="valid",
                      input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu", padding="valid"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu", padding="valid"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu", padding="valid"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

input_shape = X_train.shape[1:]   # (128, 128, 3)
model = build_alt_lenet(input_shape)
model.summary()

# ---------- 4. Train ----------
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,          # you can bump to 20 if you have time
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print("Alt-LeNet-5 (new env) - Test Accuracy:", test_acc)

# ---------- 5. Save model ----------
model.save("damage.keras")
print("Saved model to damage.keras")
