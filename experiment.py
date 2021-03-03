import argparse
import pathlib
import time
from os import listdir
from random import sample

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img

current_working_path = pathlib.Path().absolute()


def preprocess_image(image):

    image = img_to_array(image)
    image = preprocess_input(image)

    return image


def load_dataset():
    dataset = []
    for image in listdir("%s/data" % current_working_path):
        dataset.append(
            preprocess_image(
                load_img(
                    "%s/data/%s" % (current_working_path, image), target_size=(224, 224)
                )
            )
        )

    return dataset


def prediction(model, dataset):
    for run_index in range(20):
        start_time = time.time()
        samples = sample(dataset, 32)
        with tf.device(device):
            model.predict(np.stack(dataset))
        prediction_time = time.time() - start_time

        yield prediction_time


def experiment(is_nchw=False):
    if is_nchw:
        data_format = "channels_first"
        col_name = "nchw"
    else:
        data_format = "channels_last"
        col_name = "nhwc"
    tf.keras.backend.set_image_data_format(data_format)
    dataset = load_dataset()
    model = ResNet50()
    prediction_times = prediction(model, dataset)

    return pd.DataFrame(prediction_times, columns=[col_name])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", default="gpu")

    args = parser.parse_args()

    device = "/%s:0" % args.device if args.device == "gpu" else args.device

    nchw = experiment(is_nchw=True)

    nhwc = experiment()
    all_df = nhwc.merge(nchw, left_index=True, right_index=True)
    print(all_df.to_markdown())
