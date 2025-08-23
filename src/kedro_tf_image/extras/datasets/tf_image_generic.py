from pathlib import PurePosixPath
from typing import Any, Dict, Optional
from copy import deepcopy


from kedro.io.core import (
    AbstractVersionedDataset,
    Version,
    get_filepath_str,
    get_protocol_and_path,
)

import fsspec
import numpy as np

# for loading/processing the images
from keras.preprocessing.image import load_img
from keras.preprocessing.image import save_img
from keras.applications.resnet50 import preprocess_input

class TfImageGeneric(AbstractVersionedDataset):
    """``TfGenericImage`` loads / save image data from a given filepath as `numpy` array using Pillow.

    @deprecated: Use :func:`kedro_tf_image.extras.datasets.tf_image_dataset.TfImageDataSet` instead.
    Example:
    ::

        >>> TfGenericImage(filepath='/img/file/path.png')
    """


    DEFAULT_SAVE_ARGS = {}
    DEFAULT_LOAD_ARGS = {
    }

    def __init__(
        self,
        filepath: str,
        imagedim: int = 224,
        load_args: Optional[Dict[str, Any]] = None,
        save_args: Optional[Dict[str, Any]] = None,
        version: Optional[Version] = None,
    ):
        """Creates a new instance of ImageDataSet to load / save image data for given filepath.

        Args:
            filepath: The location of the image file to load / save data.
            imagedim: Target image dimension (assumes square images).
            load_args: Extra arguments for keras.preprocessing.image.load_img.
            save_args: Extra arguments for keras.preprocessing.image.save_img.
            version: Optional Kedro dataset Version.
        """
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._fs = fsspec.filesystem(self._protocol)
        super().__init__(PurePosixPath(path), version=version)  # sets _filepath, _version, _version_cache
        self._imagedim = imagedim
        self._load_args = deepcopy(self.DEFAULT_LOAD_ARGS)
        if load_args:
            self._load_args.update(load_args)
        self._save_args = deepcopy(self.DEFAULT_SAVE_ARGS)
        if save_args:
            self._save_args.update(save_args)

    def _load(self) -> np.ndarray:
        """Loads data from the image file.

        Returns:
            Data from the image file as a numpy array
        """
        # using get_filepath_str ensures that the protocol and path are appended correctly for different filesystems
        load_path = get_filepath_str(self._get_load_path(), self._protocol)
        img = load_img(load_path, **self._load_args)
        np_image = np.array(img)
        # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
        reshaped_img = np_image.reshape(
            1, self._imagedim, self._imagedim, 3)
        # prepare image for model
        imgx = preprocess_input(reshaped_img)
        # get the feature vector
        return np.asarray(imgx)

    def _save(self, data: np.ndarray) -> None:
        """Saves image data to the specified filepath."""
        # using get_filepath_str ensures that the protocol and path are appended correctly for different filesystems
        save_path = get_filepath_str(self._get_save_path(), self._protocol)
        save_img(save_path, data[0], **self._save_args)

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset."""
        return dict(filepath=self._filepath, protocol=self._protocol)
