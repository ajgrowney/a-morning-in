"""
This is the portion of the google notebook
dedicated to building and training the model
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tiktoken

RNN_UNITS = 1024
MAX_LEN = 80
DEFAULT_STATE = lambda x: tf.zeros(shape=(x,RNN_UNITS), dtype=tf.float32)

def load_dataset(path_to_file:str, seq_length = 100):
    """
    :return: dataset, ids_from_chars, chars_from_ids
    """
    text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
    print(text)
    encoder = tiktoken.get_encoding("r50k_base")
    all_ids = encoder.encode(text)
    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
    sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)
    def split_input_target(sequence):
        input_text = sequence[:-1]
        target_text = sequence[1:]
        return input_text, target_text
    dataset = sequences.map(split_input_target)
    return (dataset, encoder)

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

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    if states is None:
       states = DEFAULT_STATE(inputs.shape[0])
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    else:
      return x

class TextGenerator(keras.callbacks.Callback):
    """A callback to generate text from a trained model.
    1. Feed some starting prompt to the model
    2. Predict probabilities for the next token
    3. Sample the next token and add it to the next input

    Arguments:
        max_tokens: Integer, the number of tokens to be generated after prompt.
        start_tokens: List of integers, the token indices for the starting prompt.
        index_to_word: List of strings, obtained from the TextVectorization layer.
        top_k: Integer, sample from the `top_k` token predictions.
    """

    def __init__(
        self, max_tokens, start_tokens, index_to_word, top_k=10
    ):
        self.max_tokens = max_tokens
        self.start_tokens = start_tokens
        self.index_to_word = index_to_word
        self.k = top_k

    def sample_from(self, logits):
        logits, indices = tf.math.top_k(logits, k=self.k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)

    def detokenize(self, number):
        return self.index_to_word[number]

    def on_epoch_end(self, epoch, logs=None):
        start_tokens = [_ for _ in self.start_tokens]
        num_tokens_generated = 0
        tokens_generated = []
        while num_tokens_generated <= self.max_tokens:
            pad_len = MAX_LEN - len(start_tokens)
            sample_index = len(start_tokens) - 1
            if pad_len < 0:
                x = start_tokens[:MAX_LEN]
                sample_index = MAX_LEN - 1
            elif pad_len > 0:
                x = start_tokens + [0] * pad_len
            else:
                x = start_tokens
            x = np.array([x])
            y, _ = self.model.predict(x)
            sample_token = self.sample_from(y[0][sample_index])
            tokens_generated.append(sample_token)
            start_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)
        txt = " ".join(
            [self.detokenize(_) for _ in self.start_tokens + tokens_generated]
        )
        print(f"generated text:\n{txt}\n")

if __name__ == "__main__":
    model_id = "logic_v2"
    lyrics_file = 'data/Logic_lyrics.txt'
    # ---- Dataset Selection
    dataset, encoder = load_dataset(lyrics_file)
    vocab_size = encoder.max_token_value
    train_batches = create_training_batches(dataset)

    # ---- Model Configuration
    embedding_dim = 256
    model = MyModel(vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        rnn_units=RNN_UNITS)
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss)
    # ---- Fitting ---- 

    # Directory where the checkpoints will be saved
    checkpoint_dir = f'./training_checkpoints/{model_id}'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix, save_weights_only=True)
    
    EPOCHS = 1
    history = model.fit(train_batches, epochs=EPOCHS, callbacks=[checkpoint_callback])

    # ---- Saving ----
    with open(f"models/{model_id}.json","w") as f:
      f.write(model.to_json())
    model.save_weights(f'models/{model_id}.h5')
