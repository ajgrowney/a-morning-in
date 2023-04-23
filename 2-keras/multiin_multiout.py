import json
from keras.layers import Input, Dense
from keras.models import Model
# Multi in, multi out
from keras.layers import concatenate

mimo_x_in = Input(shape=(100,), name="x_in")
mimo_y_in = Input(shape=(100,), name = "y_in")
mimo_x = Dense(64, activation="relu")(mimo_x_in)
mimo_y = Dense(64, activation="relu")(mimo_y_in)
mimo_z = concatenate([mimo_x, mimo_y])

mimo_out_1 = Dense(1, activation="sigmoid", name="out_1")(mimo_z)
mimo_out_2 = Dense(10, activation="softmax", name="out_2")(mimo_z)

model = Model(inputs=[mimo_x_in, mimo_y_in], outputs=[mimo_out_1, mimo_out_2])
print(model.summary())

model.compile(
    optimizer="rmsprop",
    loss = {
        "out_1": "binary_crossentropy",
        "out_2": "categorical_crossentropy" },
    loss_weights = {
        "out_1": 1.,
        "out_2": 0.2})

# Sample Input
import numpy as np
from keras.utils import to_categorical
data = np.random.random((1000,100))
xs = np.random.randint(2, size = (1000,1))
ys = np.random.randint(10, size = (1000,1))

history = model.fit(
    x = { "x_in": data, "y_in": data},
    y = {"out_1": xs, "out_2": to_categorical(ys)},
    epochs=1, batch_size=32)
print(json.dumps(model.get_config(), indent=2))
print(json.dumps(history.history, indent=2))