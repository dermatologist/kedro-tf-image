# Kedro TF Image :framed_picture:

This package consists of [Kedro pipelines](https://kedro.readthedocs.io/en/stable/kedro.pipeline.html) for preprocessing images using TensorFlow. I use it for [Dermatology workflows](https://skinhelpdesk.com) and [multimodal machine learning](https://github.com/dermatologist/kedro-multimodal). Use [this template](https://github.com/dermatologist/kedro-multimodal) that imports this package for multimodal ML. This package includes [Kedro datasets](https://kedro.readthedocs.io/en/stable/kedro.extras.datasets.html) for [loading weights](src/kedro_tf_image/extras/datasets/tf_model_weights.py) (as in CheXnet weights to a DenseNet121) and [downloading pre-trained models](src/kedro_tf_image/extras/datasets/tf_model_download.py) from TF hub.

## [Pipelines](src/kedro_tf_image/pipelines/preprocess/pipeline.py)
- The **download** pipeline downloads online images defined in a csv file for multilabel classification. The labels are added to the filename. The csv format is:

```
id, url, labels
1, https://somesite.com/someimage.jpg,dog|black|grey
```

- The **folder** pipeline creates TensorFlow dataset from a folder of images with labels as subfolders.
- The **multilabel** pipeline processes files downloaded by the 'download' pipeline and create a dataset with images and labels. The labels are extracted from the filename. Example: _dog_black.jpg
- Add labels in [parameters.yml](conf/base/parameters/preprocess.yml)

```
master_labels: ["cat", "dog", "white", "black", "tan"]
val_size: 0.2
```


## How to install

- pip install git+https://github.com/dermatologist/kedro-tf-image.git

## How to use

```
from kedro_tf_image.pipelines import preprocess

download = preprocess.create_download_pipeline(
        input="csvdata", output="imageset") #input is csv
folder = preprocess.create_folder_pipeline(
        input="imagefolder", output="processeddataset")
multilabel = preprocess.create_multilabel_pipeline(input="imageset", output="processeddataset")

# check output keys in the catalog below
```


## [Catalog](conf/base/catalog.yml)


```

imageset:
  type: PartitionedDataSet
  dataset: {
      "type": "kedro_tf_image.extras.datasets.tf_image_dataset.TfImageDataSet",
      "imagedim": 224,
      "preprocess_input": "tensorflow.keras.applications.resnet50.preprocess_input"
  }
  path: data/01_raw/imageset
  filename_suffix: ".jpg"

csvdata:
  type: pandas.CSVDataSet
  filepath: data/01_raw/csvfile.csv

imagefolder:
  type: kedro_tf_image.extras.datasets.tf_image_folder.TfImageFolder
  folderpath: "/path/to/images"
  imagedim: 224
  load_args:
    validation_split: 0.2
    seed: 123
    batch_size: 1


processeddataset:
  type: kedro_tf_image.extras.datasets.tf_image_processed.TfImageProcessed
  folderpath: data/02_intermediate/
  imagedim: 224

# This is required as copy_mode: assign is needed for TF datasets
datasetinmemory:
  type: MemoryDataSet
  copy_mode: assign

```

## [Datasets](src/kedro_tf_image/extras/datasets/)

* kedro_tf_image.extras.datasets.tf_image_dataset.TfImageDataSet - Load single images
* kedro_tf_image.extras.datasets.tf_image_folder.TfImageFolder - Load a folder of images
* kedro_tf_image.extras.datasets.tf_model_weights.TfModelWeights - Read model from weights (Ex: CheXnet with dim 14)
(Use [create_classification_layer](src/kedro_tf_image/pipelines/preprocess/pipeline.py) to add a Dense layer of NCLASSES dim)
* kedro_tf_image.extras.datasets.tf_model_download.TfModelDownload - Load model from TF hub.
## Author

- [Bell Eapen](https://nuchange.ca) [![Twitter Follow](https://img.shields.io/twitter/follow/beapen?style=social)](https://twitter.com/beapen)

