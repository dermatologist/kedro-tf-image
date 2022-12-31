import importlib
import os
from typing import Any, Dict, Tuple
from copy import deepcopy

from kedro.io.core import (
    AbstractVersionedDataSet,
)
import tensorflow as tf
from cachetools import Cache
import importlib
from keras.layers import Input
from keras.layers.core import Dense
from keras.models import Model

class TfModelWeights(AbstractVersionedDataSet):

    """_summary_
    _description_

    _version_
    architecture: str (e.g. "VGG16")
    filepath: str (e.g. "path/to/weights.h5")
    """
    # class_num=2, input_shape=None, use_base_weights=True
    DEFAULT_LOAD_ARGS = {
        "class_num": 14,
        "input_shape": None,
        "use_base_weights": True,
    }  # type: Dict[str, Any]
    DEFAULT_SAVE_ARGS = {}  # type: Dict[str, Any]
    def __init__(self, filepath: str, architecture: str = "DenseNet121", load_args: Dict[str, Any] = None, save_args: Dict[str, Any] = None):
        self._version = None
        self._version_cache = Cache(maxsize=2)
        self._filepath = filepath
        self._architecture = architecture
        # Handle default load arguments
        self._load_args = deepcopy(self.DEFAULT_LOAD_ARGS)
        if load_args is not None:
            self._load_args.update(load_args)
        ## Default models
        self._models = dict(
                VGG16=dict(
                    input_shape=(224, 224, 3),
                    module_name="vgg16",
                    last_conv_layer="block5_conv3",
                ),
                VGG19=dict(
                    input_shape=(224, 224, 3),
                    module_name="vgg19",
                    last_conv_layer="block5_conv4",
                ),
                DenseNet121=dict(
                    input_shape=(224, 224, 3),
                    module_name="densenet",
                    last_conv_layer="bn",
                ),
                ResNet50=dict(
                    input_shape=(224, 224, 3),
                    module_name="resnet50",
                    last_conv_layer="activation_49",
                ),
                InceptionV3=dict(
                    input_shape=(299, 299, 3),
                    module_name="inception_v3",
                    last_conv_layer="mixed10",
                ),
                InceptionResNetV2=dict(
                    input_shape=(299, 299, 3),
                    module_name="inception_resnet_v2",
                    last_conv_layer="conv_7b_ac",
                ),
                NASNetMobile=dict(
                    input_shape=(224, 224, 3),
                    module_name="nasnet",
                    last_conv_layer="activation_188",
                ),
                NASNetLarge=dict(
                    input_shape=(331, 331, 3),
                    module_name="nasnet",
                    last_conv_layer="activation_260",
                ),
            )

        # Handle default save arguments
        self._save_args = deepcopy(self.DEFAULT_SAVE_ARGS)
        if save_args is not None:
            self._save_args.update(save_args)

    def _load(self) -> Tuple:
        model = self.get_model(self._filepath, self._architecture, **self._load_args)
        return model

    def _save(self, model: Any) -> None:
        model.save_weights(self._filepath)

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset."""
        return dict(filepath=self._filepath, architecture=self._architecture)

    # https://www.kaggle.com/code/ashishpatel26/chexnet-radiologist-level-pneumonia-detection/notebook?utm_source=pocket_saves
    def get_model(self, weights_path=None, model_name="DenseNet121", **kwargs):

        # class_num=2, input_shape=None, use_base_weights=True
        class_num = kwargs.get("class_num", 14)
        input_shape = kwargs.get("input_shape", None)
        use_base_weights = kwargs.get("use_base_weights", True)

        if use_base_weights is True:
            base_weights = "imagenet"
        else:
            base_weights = None

        base_model_class = getattr(
            importlib.import_module(
                f"keras.applications.{self._models[model_name]['module_name']}"
            ),
            model_name)

        if input_shape is None:
            input_shape = self._models[model_name]["input_shape"]

        img_input = Input(shape=input_shape)

        base_model = base_model_class(
            include_top=False,
            input_tensor=img_input,
            input_shape=input_shape,
            weights=base_weights,
            pooling="avg")
        x = base_model.output
        predictions = Dense(class_num, activation="sigmoid",
                            name="predictions")(x)
        model = Model(inputs=img_input, outputs=predictions)

        if weights_path == "":
            weights_path = None

        if weights_path is not None:
            print(f"load model weights_path: {weights_path}")
            model.load_weights(weights_path)
        return model
