"""
Loading and using the model to generate text
"""
import tensorflow as tf

if __name__ == "__main__":
    model_id = 'logic_v2'
    one_step_reloaded = tf.saved_model.load(f'models/{model_id}')
    states = None
    next_char = tf.constant(['Testing'])
    result = [next_char]

    for n in range(1000):
        next_char, states = one_step_reloaded.generate_one_step(next_char, states=states)
        result.append(next_char)
        print("-"*64)
        print(tf.strings.join(result)[0].numpy().decode("utf-8"))