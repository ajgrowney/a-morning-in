from keras.layers import Input, Dense
from keras.models import Model

inputs = Input(shape=(256,))
x = Dense(64, activation="relu")(inputs)
x = Dense(64, activation="relu")(x)
predictions = Dense(10, activation="softmax")(x)

model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics= ["accuracy"])

