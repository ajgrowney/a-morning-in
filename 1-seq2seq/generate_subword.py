
"""
Loading and using the model to generate text
"""
import numpy as np
import tiktoken
import tensorflow as tf

if __name__ == "__main__":
    # Select Encoder
    encoder = tiktoken.get_encoding("p50k_base")

    # Load Model
    one_step_reloaded = tf.saved_model.load(f'models/logic_v3')
    states = one_step_reloaded.get_initial_state()

    # Initialize
    init_words = 'Hey'
    init_ids = encoder.encode(init_words)
    result = init_ids
    next_ids = np.array(init_ids)
    for n in range(100):
        next_ids = np.array(next_ids).reshape(1,-1)
        next_ids, states = one_step_reloaded.generate_one_step(next_ids, states=states)
        result.extend(next_ids)
    print(len(result))
    print(encoder.decode(result))