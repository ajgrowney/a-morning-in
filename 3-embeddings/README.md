# Goal

From where we previously left off with the RNN, we are taking in 100 characters as input and producing the next character.
We are using the `tf.keras.layers.StringLookup` as input to the embedding layer and I want to explore other alternatives in this space.

## What I learned
1. You can make use of layers individually which is wild
Ex: `model.embeddings(input)`

