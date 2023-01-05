from pathlib import PurePosixPath
from typing import Any, Callable, Dict
from copy import deepcopy

from kedro.io.core import (
    PROTOCOL_DELIMITER,
    AbstractVersionedDataSet,
    DataSetError,
    Version,
    get_filepath_str,
    get_protocol_and_path,
)

import fsspec
import numpy as np

# for loading/processing the images
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import save_img
from kedro.utils import load_obj
from kedro.extras.datasets.pillow.image_dataset import ImageDataSet
from tensorflow.keras.applications.resnet50 import preprocess_input
# The following is supplied as callable and defined in the catalog.yml
"""
imageset:
  type: PartitionedDataSet
  dataset: {
      "type": "kedro_tf_image.extras.datasets.tf_image_dataset.TfImageDataSet",
      "preprocess_input": "tensorflow.keras.applications.resnet50.preprocess_input"
  }
  path: data/01_raw/imageset
  filename_suffix: ".jpg"
"""
# from tensorflow.keras.applications.resnet50 import preprocess_input
class TfImageDataSet(AbstractVersionedDataSet):
    """``TfImageDataSet`` loads / save image data from a given filepath as `numpy` array using TF.

    Differences from ImageDataSet
        * uses tf functions to load, save and process images

    Example:
    ::

        >>> TfImageDataSet(filepath='/img/file/path.png', preprocess_input=tensorflow.keras.applications.resnet50.preprocess_input)
    """

    DEFAULT_SAVE_ARGS = {}  # type: Dict[str, Any]


    # These should be supplied in the catalog
    def __init__(self,
                 filepath: str,
                 preprocess_input: str,  # defaults to ResNet50 preprocessor
                 save_args: Dict[str, Any] = None,
                 version: Version = None,
                 credentials: Dict[str, Any] = None,
                 fs_args: Dict[str, Any] = None,
                 imagedim: int = 224): #defaults to ResNet50 dim
        """Creates a new instance of ImageDataSet to load / save image data for given filepath.

        Args:
            filepath: The location of the image file to load / save data.
        """
        _fs_args = deepcopy(fs_args) or {}
        _fs_open_args_load = _fs_args.pop("open_args_load", {})
        _fs_open_args_save = _fs_args.pop("open_args_save", {})
        _credentials = deepcopy(credentials) or {}

        protocol, path = get_protocol_and_path(filepath, version)
        if protocol == "file":
            _fs_args.setdefault("auto_mkdir", True)

        self._protocol = protocol
        self._fs = fsspec.filesystem(self._protocol, **_credentials, **_fs_args)

        super().__init__(
            filepath=PurePosixPath(path),
            version=version,
            exists_function=self._fs.exists,
            glob_function=self._fs.glob,
        )

        # Handle default save argument
        self._save_args = deepcopy(self.DEFAULT_SAVE_ARGS)
        if save_args is not None:
            self._save_args.update(save_args)

        _fs_open_args_save.setdefault("mode", "wb")
        self._fs_open_args_load = _fs_open_args_load
        self._fs_open_args_save = _fs_open_args_save
        self._preprocess_input = load_obj(
            preprocess_input, "tensorflow.keras.applications.resnet50.preprocess_input") # second argument is the default
        self._imagedim = imagedim

    def _load(self) -> np.ndarray:
        """Loads data from the image file.

        Returns:
            Data from the image file as a numpy array
        """
        load_path = str(self._get_load_path())
        if self._protocol != "file":
            # file:// protocol seems to misbehave on Windows
            # (<urlopen error file not on local host>),
            # so we don't join that back to the filepath;
            # storage_options also don't work with local paths
            load_path = f"{self._protocol}{PROTOCOL_DELIMITER}{load_path}"

        img = load_img(load_path, target_size=(self._imagedim, self._imagedim))
        np_image = np.array(img)
        # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
        reshaped_img = np_image.reshape(1, self._imagedim, self._imagedim, 3)
        # prepare image for model
        imgx = self._preprocess_input(reshaped_img)
        # get the feature vector
        return np.asarray(imgx)

    def _save(self, data: np.ndarray) -> None:
        """Saves image data to the specified filepath."""
        # using get_filepath_str ensures that the protocol and path are appended correctly for different filesystems
        save_path = get_filepath_str(self._get_save_path(), self._protocol)
        reshaped_img = data.reshape(self._imagedim, self._imagedim, 3)  # (224,224,3)
        save_img(save_path, reshaped_img)

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset."""
        return dict(filepath=self._filepath, protocol=self._protocol, preprocess_input=self._preprocess_input)


    def _exists(self) -> bool:
        try:
            load_path = get_filepath_str(self._get_load_path(), self._protocol)
        except DataSetError:
            return False

        return self._fs.exists(load_path)

    def _release(self) -> None:
        super()._release()
        self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        """Invalidate underlying filesystem caches."""
        filepath = get_filepath_str(self._filepath, self._protocol)
        self._fs.invalidate_cache(filepath)