import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json
import tiktoken
from train import MyModel

MAX_LEN = 100
RNN_UNITS = 1024

class OneStep(tf.keras.Model):
    def __init__(self, model, encoder, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.encoder = encoder
    
    def get_initial_state(self):
        """Generate initial state for the GRU layer"""
        return tf.zeros(shape=(1,RNN_UNITS), dtype=tf.float32)

    def clean_input(self, tokens):
        pad_len = MAX_LEN - len(tokens)
        sample_index = len(tokens) - 1
        if pad_len < 0:
            x = tokens[:MAX_LEN]
            sample_index = MAX_LEN - 1
        elif pad_len > 0:
            x = tokens + [0] * pad_len
        else:
            x = tokens
        x = np.array([x])
        return x

    def generate_one_step(self, input_data:str, states):
        # Convert strings to token IDs.
        print(f"AHHH: {input_data}")
        input_ids = self.clean_input(self.encoder.encode(input_data))

        # Run the model.
        # predicted_logits.shape is [batch, char, next_char_logits]
        predicted_logits, states = self.model(inputs=input_ids, states=states,
                                            return_state=True)
        # Only use the last prediction.
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits/self.temperature

        # Sample the output logits to generate token IDs.
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        # Convert from token ids to characters
        predicted_tokens = self.encoder.decode(predicted_ids)

        # Return the tokens and model state.
        return predicted_tokens, states

if __name__ == "__main__":
    enc = tiktoken.get_encoding("r50k_base")
    with open("models/logic_v2.json","r") as f:
        model_json = f.read()
        model = model_from_json(model_json)
    model.load_weights("models/logic_v2.h5")
    text_gen = OneStep(model, enc)
    states = text_gen.get_initial_state()
    result = "Hello"

    for n in range(10):
        print(f"Result: {result}")
        print(f"States: {states}")
        next_tokens, states = text_gen.generate_one_step(result, states=None)
        result += " ".join(next_tokens)
        print("-"*64)
        print(result)
