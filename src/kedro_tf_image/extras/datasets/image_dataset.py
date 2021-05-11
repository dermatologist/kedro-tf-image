import os
from pathlib import PurePosixPath
from typing import Any, Callable, Dict

from kedro.io.core import (
    get_filepath_str,
    get_protocol_and_path,
)

import fsspec
import numpy as np

# for loading/processing the images
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import save_img
from kedro.extras.datasets.pillow.image_dataset import ImageDataSet
from tensorflow.keras.applications.resnet50 import preprocess_input
# The following is supplied as callable and defined in the catalog.yml
"""
imageset:
  type: PartitionedDataSet
  dataset: {
      "type": "kedro_tf_image.extras.datasets.image_dataset.TfImageDataSet",
      "preprocess_input": "tensorflow.keras.applications.resnet50.preprocess_input"
  }
  path: data/01_raw/imageset
  filename_suffix: ".jpg"
"""
# from tensorflow.keras.applications.resnet50 import preprocess_input
class TfImageDataSet(ImageDataSet):
    """``ImageDataSet`` loads / save image data from a given filepath as `numpy` array using Pillow.

    Differences from parent class
        * uses tf functions to load, save and process images
        * uses folderpath instead of filepath for save
        * filename should be supplied along with save
        * save accepts a Dict as below:
        {
            image: np.ndarray,
            filename: str
        }

    Example:
    ::

        >>> TfImageDataSet(filepath='/img/file/path.png', preprocess_input=tensorflow.keras.applications.resnet50.preprocess_input)
    """

    # These should be supplied in the catalog
    def __init__(self, filepath: str,
                preprocess_input: Callable = preprocess_input, #defaults to ResNet50 preprocessor
                imagedim: int = 224): #defaults to ResNet50 dim
        """Creates a new instance of ImageDataSet to load / save image data for given filepath.

        Args:
            filepath: The location of the image file to load / save data.
        """
        # parse the path and protocol (e.g. file, http, s3, etc.)
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol)
        self._version = None
        self._preprocess_input = preprocess_input
        self._imagedim = imagedim

    def _load(self) -> np.ndarray:
        """Loads data from the image file.

        Returns:
            Data from the image file as a numpy array
        """
        # using get_filepath_str ensures that the protocol and path are appended correctly for different filesystems
        load_path = get_filepath_str(self._get_load_path(), self._protocol)
        img = load_img(load_path, target_size=(self._imagedim, self._imagedim))
        np_image = np.array(img)
        # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
        reshaped_img = np_image.reshape(1, self._imagedim, self._imagedim, 3)
        # prepare image for model
        imgx = self._preprocess_input(reshaped_img)
        # get the feature vector
        return np.asarray(imgx)

    def _save(self, data: Dict[str, Any]) -> None:
        """Saves image data to the specified filepath."""
        # using get_filepath_str ensures that the protocol and path are appended correctly for different filesystems
        save_path = get_filepath_str(self._get_save_path(), self._protocol)
        save_folder = os.path.dirname(save_path)
        save_img(save_folder + data['filename'], data['image'])

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset."""
        return dict(filepath=self._filepath, protocol=self._protocol, preprocess_input=self._preprocess_input)
