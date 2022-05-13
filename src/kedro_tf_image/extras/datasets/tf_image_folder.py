from typing import Any, Dict, Tuple
from copy import deepcopy

from kedro.io.core import (
    AbstractVersionedDataSet,
)
from tensorflow.keras.preprocessing import image_dataset_from_directory

class TfImageFolder(AbstractVersionedDataSet):
    """``ImageDataSet`` loads / save image data from a given folder
    Example:
    ::

        >>> TfImageFolder(filepath='/img/file/')


    path/to/image_dir/
    split_name/  # Ex: 'train'
        label1/  # Ex: 'airplane' or '0015'
        xxx.png
        xxy.png
        xxz.png
        label2/
        xxx.png
        xxy.png
        xxz.png
    split_name/  # Ex: 'test'
        ...
    """
    DEFAULT_LOAD_ARGS = {

    }  # type: Dict[str, Any]

    def __init__(self, folderpath: str, imagedim: int, load_args: Dict[str, Any]):
        self._version = None
        self._folderpath = folderpath
        self._imagedim = imagedim
        # Handle default load arguments
        self._load_args = deepcopy(self.DEFAULT_LOAD_ARGS)
        if load_args is not None:
            self._load_args.update(load_args)

    def _load(self) -> Any:
        train_ds = image_dataset_from_directory(
            self._folderpath,
            image_size=(self._imagedim, self._imagedim),
            subset = 'training',
            **self._load_args)
        val_ds = image_dataset_from_directory(
            self._folderpath,
            image_size=(self._imagedim, self._imagedim),
            subset='validation',
            **self._load_args)
        return(train_ds, val_ds)

    def _save(self, data: Any) -> None:
        """Not implemented."""
        pass

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset."""
        return dict(folderpath=self._folderpath, load_args=self._load_args)
