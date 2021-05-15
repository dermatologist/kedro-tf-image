import os
from typing import Any, Dict, Tuple
from copy import deepcopy

from kedro.io.core import (
    AbstractVersionedDataSet,
)
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow as tf
from cachetools import Cache


class TfImageProcessed(AbstractVersionedDataSet):

    DEFAULT_LOAD_ARGS = {

    }  # type: Dict[str, Any]

    DEFAULT_SAVE_ARGS = {

    }  # type: Dict[str, Any]

    def __init__(self, folderpath: str, imagedim: int = 224, load_args: Dict[str, Any] = None, save_args: Dict[str, Any] = None):
        self._version = None
        self._version_cache = Cache(maxsize=2)
        self._folderpath = folderpath
        # Handle default load arguments
        self._load_args = deepcopy(self.DEFAULT_LOAD_ARGS)
        if load_args is not None:
            self._load_args.update(load_args)
        if "element_spec" not in self._load_args:
            spec = (tf.TensorSpec(shape=(None, imagedim, imagedim, 3), dtype=tf.float32, name=None),
                    tf.TensorSpec(shape=(None,), dtype=tf.int32, name=None))
            self._load_args['element_spec'] = spec
        # Handle default save arguments
        self._save_args = deepcopy(self.DEFAULT_SAVE_ARGS)
        if save_args is not None:
            self._save_args.update(save_args)

    def _load(self) -> Tuple:
        train_ds_path = os.path.join(self._folderpath, "train_ds")
        train_ds = tf.data.experimental.load(
            train_ds_path, **self._load_args
        )
        val_ds_path = os.path.join(self._folderpath, "val_ds")
        val_ds = tf.data.experimental.load(
            val_ds_path, **self._load_args
        )
        return(train_ds, val_ds)

    def _save(self, data: Tuple) -> None:
        (train_ds, val_ds) = data
        train_ds_path = os.path.join(self._folderpath, "train_ds")
        tf.data.experimental.save(
            train_ds, train_ds_path, **self._save_args
        )
        val_ds_path = os.path.join(self._folderpath, "val_ds")
        tf.data.experimental.save(
            val_ds, val_ds_path, **self._save_args
        )

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset."""
        return dict(folderpath=self._folderpath)
