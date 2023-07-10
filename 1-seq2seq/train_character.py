"""
This is the portion of the google notebook
dedicated to building and training the model
"""
import os
import tensorflow as tf

RNN_UNITS = 1024
def load_dataset(text:str, encoder):
    """
    :return: dataset
    """
    all_ids = encoder(tf.strings.unicode_split(text, 'UTF-8'))
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
    x         = inputs
    x         = self.embedding(x, training=training)
    x, states = self.gru(x, initial_state=states, training=training)
    x         = self.dense(x,     training=training)

    if return_state:
      return x, states
    else:
      return x

class OneStep(tf.keras.Model):
  def __init__(self, model, decoder, encoder, prediction_mask = None, temperature=1.0):
    super().__init__()
    self.temperature = temperature
    self.model = model
    self.decoder = decoder
    self.encoder = encoder
    self.prediction_mask = prediction_mask
    
  @tf.function(input_signature=[tf.TensorSpec(shape=[1], dtype=tf.string)])
  def get_initial_state(self, inputs):
    """Generate initial state for the GRU layer"""
    return tf.zeros(shape=(1,RNN_UNITS), dtype=tf.float32)

  @tf.function(input_signature=[tf.TensorSpec(shape=[1], dtype=tf.string), tf.TensorSpec(shape=(1, RNN_UNITS), dtype=tf.float32)])
  def generate_one_step(self, inputs, states=None):
    # Convert strings to token IDs.
    input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
    input_ids = self.encoder(input_chars).to_tensor()

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

    # Convert from token ids to characters
    predicted_chars = self.decoder(predicted_ids)

    # Return the characters and model state.
    return predicted_chars, states
    
if __name__ == "__main__":
    # ---- Dataset Loading
    model_id = "logic_v2"
    lyrics_file = 'data/Logic_lyrics.txt'
    text = open(lyrics_file, 'rb').read().decode(encoding='utf-8')
    vocab = sorted(set(text))

    # ---- Dataset Formatting
    encoder = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)
    decoder = tf.keras.layers.StringLookup(vocabulary=encoder.get_vocabulary(), invert=True, mask_token=None)

    dataset = load_dataset(text, encoder)
    
    train_batches = create_training_batches(dataset)

    # ---- Model Configuration
    vocab_size = len(encoder.get_vocabulary())
    embedding_dim = 256
    model = MyModel(vocab_size=vocab_size,
        embedding_dim=embedding_dim, rnn_units=RNN_UNITS)
    model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True))

    # ---- Fitting ---- 

    # Directory where the checkpoints will be saved
    checkpoint_dir = f'./training_checkpoints/{model_id}'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix, save_weights_only=True)
    
    EPOCHS = 300
    history = model.fit(train_batches, epochs=EPOCHS, callbacks=[checkpoint_callback])

    # ---- "OneStep" Model
    # Create the prediction mask to prevent "[UNK]" from being generated.
    skip_ids = encoder(['[UNK]'])[:, None]
    sparse_mask = tf.SparseTensor(
        # Put a -inf at each bad index.
        values=[-float('inf')]*len(skip_ids), indices=skip_ids,
        # Match the shape to the vocabulary
        dense_shape=[len(encoder.get_vocabulary())])
    prediction_mask = tf.sparse.to_dense(sparse_mask)
    one_step_model = OneStep(model, decoder, encoder, prediction_mask)

    # ---- Saving ----
    tf.saved_model.save(one_step_model, f'models/{model_id}')
