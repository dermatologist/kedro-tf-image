# Copyright 2021 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Ref: https://stackoverflow.com/questions/60430277/how-to-load-images-from-url-with-a-tensorflow-2-dataset
Ref2: https://www.tensorflow.org/tutorials/load_data/images
"""


from typing import Any, Dict
import tensorflow as tf
import numpy as np
import cv2
from urllib.request import urlopen
import matplotlib.pyplot as plt
import pandas as pd

def get(url):
    with urlopen(str(url.numpy().decode("utf-8"))) as request:
        img_array = np.asarray(bytearray(request.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_image_from_url(url):
    return tf.py_function(get, [url], tf.uint8)


def load_data_from_url(data: pd.DataFrame) -> None:
    print(data)

"""
dataset = tf.data.Dataset.from_tensor_slices(image_urls)

def get(url):
    with urlopen(str(url.numpy().decode("utf-8"))) as request:
        img_array = np.asarray(bytearray(request.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def read_image_from_url(url):
    return tf.py_function(get, [url], tf.uint8)


dataset_images = dataset.map(lambda x: read_image_from_url(x))

for d in dataset_images:
  print(d)


"""
