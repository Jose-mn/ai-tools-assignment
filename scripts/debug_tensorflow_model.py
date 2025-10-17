# debug_tensorflow_model.py
"""
Task 4 - AI Debugging & Ethics
------------------------------
This script demonstrates identifying and fixing a TensorFlow model bug,
and reflects on ethical practices in AI development.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Generate dummy data
X = np.random.rand(100, 3)
y = (np.sum(X, axis=1) > 1.5).astype(int)

# ----------------------------
# ❌ Buggy Model Definition
# ----------------------------
def buggy_model():
    model = keras.Sequential([
        layers.Dense(8, input_shape=(3,), activation='relu'),
        # Bug: Missing output activation for binary classification
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ----------------------------
# ✅ Fixed Model Definition
# ----------------------------
def fixed_model():
    model = keras.Sequential([
        layers.Dense(8, input_shape=(3,), activation='relu'),
        # Fix: Added sigmoid activation for binary output
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Try training the buggy model
print("---- Training Buggy Model ----")
try:
    buggy = buggy_model()
    buggy.fit(X, y, epochs=3, verbose=0)
    print("Buggy model trained successfully (unexpected!)")
except Exception as e:
    print("⚠️ Error found in buggy model:", e)

# Train the fixed model
print("\n---- Training Fixed Model ----")
fixed = fixed_model()
history = fixed.fit(X, y, epochs=3, verbose=0)
print("✅ Fixed model trained successfully.")
print("Final accuracy:", history.history['accuracy'][-1])

# ----------------------------
# Ethical Reflection
# ----------------------------
print("\n--- Ethical Reflection ---")
print("✅ Debugging prevents misleading AI results.")
print("✅ Ensuring proper model structure reduces bias and increases reliability.")
print("✅ Responsible AI means verifying outputs before deployment.")
