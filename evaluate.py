import numpy as np
import sys
import tensorflow as tf
from traffic import load_data


def main():

    if len(sys.argv) != 3:
        sys.exit("Usage: python evaluate.py model data_directory")

    images, labels = load_data(sys.argv[2])
    labels = tf.keras.utils.to_categorical(labels)

    model = tf.keras.models.load_model(sys.argv[1])
    model.summary()

    print("Evaluating...")
    model.evaluate(np.array(images), np.array(labels), verbose=2)


if __name__ == "__main__":
    main()
