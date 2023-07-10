"""
This is the portion of the google notebook
dedicated to building and training the model
"""
import os

import numpy as np
import tiktoken
import tensorflow as tf

RNN_UNITS = 1024
def load_dataset(text:str, encoder):
    """
    :param text: text to load into a dataset
    :return: dataset
    """
    all_ids = encoder(text)
    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
    seq_length = 100
    sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)
    def split_input_target(sequence):
        input_text = sequence[:-1]
        target_text = sequence[1:]
        return input_text, target_text
    dataset = sequences.map(split_input_target)
    return (dataset)

def create_training_batches(dataset, batch_size = 64, buffer_size = 1000):
    """
    """

    # Buffer size to shuffle the dataset
    # (TF data is designed to work with possibly infinite sequences,
    # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    # it maintains a buffer in which it shuffles elements).

    return (dataset
        .shuffle(buffer_size)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE))
    
# Model Definition
class MyModel(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units):
    super().__init__(self)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(rnn_units,
                                   return_sequences=True,
                                   return_state=True)
    self.dense = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs, states = None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    else:
      return x

class OneStep(tf.keras.Model):
  def __init__(self, model, prediction_mask = None, temperature=1.0):
    super().__init__()
    self.temperature = temperature
    self.model = model
    self.prediction_mask = prediction_mask
    
  @tf.function
  def get_initial_state(self):
    """Generate initial state for the GRU layer"""
    return tf.zeros(shape=(1,RNN_UNITS), dtype=tf.float32)

  @tf.function(input_signature=[tf.TensorSpec(shape=(1,1,), dtype=tf.float32), tf.TensorSpec(shape=(1, RNN_UNITS), dtype=tf.float32)])
  def generate_one_step(self, input_ids, states=None):
    # Run the model.
    # predicted_logits.shape is [batch, char, next_char_logits]
    predicted_logits, states = self.model(inputs=input_ids, states=states,
                                          return_state=True)
    # Only use the last prediction.
    predicted_logits = predicted_logits[:, -1, :]
    predicted_logits = predicted_logits/self.temperature
    if self.prediction_mask is not None:
      # Apply the prediction mask: prevent "[UNK]" from being generated.
      predicted_logits = predicted_logits + self.prediction_mask

    # Sample the output logits to generate token IDs.
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)

    # Return the characters and model state.
    return predicted_ids, states
    
if __name__ == "__main__":
    # ---- Dataset Loading
    model_id = "logic_v4"
    lyrics_file = 'data/Logic_lyrics.txt'
    text = open(lyrics_file, 'rb').read().decode(encoding='utf-8')
    vocab = sorted(set(text))

    # ---- Dataset Formatting
    sw_encoder = tiktoken.get_encoding("p50k_base")
    encoder = sw_encoder.encode
    decoder = sw_encoder.decode
    vocab_size = 50280

    dataset = load_dataset(text, encoder)
    
    train_batches = create_training_batches(dataset)

    # ---- Model Configuration
    embedding_dim = 256
    model = MyModel(vocab_size=vocab_size,
        embedding_dim=embedding_dim, rnn_units=RNN_UNITS)
    model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), run_eagerly = True)

    # ---- Fitting ---- 

    # Directory where the checkpoints will be saved
    checkpoint_dir = f'./training_checkpoints/{model_id}'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix, save_weights_only=True)
    
    EPOCHS = 100
    history = model.fit(train_batches, epochs=EPOCHS, callbacks=[])

    # ---- "OneStep" Model
    # Create the prediction mask to prevent "[UNK]" from being generated.
    # skip_ids = encoder(['[UNK]'])[:, None]
    # sparse_mask = tf.SparseTensor(
    #     # Put a -inf at each bad index.
    #     values=[-float('inf')]*len(skip_ids), indices=skip_ids,
    #     # Match the shape to the vocabulary
    #     dense_shape=[vocab_size])
    # prediction_mask = tf.sparse.to_dense(sparse_mask)
    one_step_model = OneStep(model, None)

    # ---- Saving ----
    tf.saved_model.save(one_step_model, f'models/{model_id}')
