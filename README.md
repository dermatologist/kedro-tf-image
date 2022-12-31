# Kedro TF Image

This package consists of [Kedro pipelines](https://kedro.readthedocs.io/en/stable/kedro.pipeline.html) for preprocessing images for TensorFlow. I use it mostly for [CNN based Dermatology workflows.](https://skinhelpdesk.com) and [multi-modal ML](https://github.com/dermatologist/kedro-tf-utils).

- The **download** pipeline downloads online images defined in a csv file for multilabel classification. The labels are added to the filename. The csv format is:

```
id, url, labels
1, https://somesite.com/someimage.jpg,dog|black|grey
```

- The **folder** pipeline creates TensorFlow dataset from a folder of images with labels as subfolders.
- The **multilabel** pipeline processes files downloaded by the 'download' pipeline and create a dataset with images and labels. The labels are extracted from the filename. Example: _dog_black.jpg
- Add labels in parameters.yml

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


## Catalog


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

## Datasets

* kedro_tf_image.extras.datasets.tf_image_dataset.TfImageDataSet - Load single images
* kedro_tf_image.extras.datasets.tf_image_folder.TfImageFolder - Load a folder of images
* kedro_tf_image.extras.datasets.tf_image_folder.TfModelWeights - Read model from weights (Ex: CheXnet)

## Author

- [Bell Eapen](https://nuchange.ca) [![Twitter Follow](https://img.shields.io/twitter/follow/beapen?style=social)](https://twitter.com/beapen)

## Overview

This is your new Kedro project, which was generated using `Kedro 0.17.3`.

Take a look at the [Kedro documentation](https://kedro.readthedocs.io) to get started.

## Rules and guidelines

In order to get the best out of the template:

- Don't remove any lines from the `.gitignore` file we provide
- Make sure your results can be reproduced by following a [data engineering convention](https://kedro.readthedocs.io/en/stable/11_faq/01_faq.html#what-is-data-engineering-convention)
- Don't commit data to your repository
- Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

## How to install dependencies

Declare any dependencies in `src/requirements.txt` for `pip` installation and `src/environment.yml` for `conda` installation.

To install them, run:

```
kedro install
```

## How to run Kedro

You can run your Kedro project with:

```
kedro run
```

## How to test your Kedro project

Have a look at the file `src/tests/test_run.py` for instructions on how to write your tests. You can run your tests as follows:

```
kedro test
```

To configure the coverage threshold, look at the `.coveragerc` file.

## Project dependencies

To generate or update the dependency requirements for your project:

```
kedro build-reqs
```

This will copy the contents of `src/requirements.txt` into a new file `src/requirements.in` which will be used as the source for `pip-compile`. You can see the output of the resolution by opening `src/requirements.txt`.

After this, if you'd like to update your project requirements, please update `src/requirements.in` and re-run `kedro build-reqs`.

[Further information about project dependencies](https://kedro.readthedocs.io/en/stable/04_kedro_project_setup/01_dependencies.html#project-specific-dependencies)

## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `context`, `catalog`, and `startup_error`.

### Jupyter

To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

### JupyterLab

To use JupyterLab, you need to install it:

```
pip install jupyterlab
```

You can also start JupyterLab:

```
kedro jupyter lab
```

### IPython

And if you want to run an IPython session:

```
kedro ipython
```

### How to convert notebook cells to nodes in a Kedro project

You can move notebook code over into a Kedro project structure using a mixture of [cell tagging](https://jupyter-notebook.readthedocs.io/en/stable/changelog.html#cell-tags) and Kedro CLI commands.

By adding the `node` tag to a cell and running the command below, the cell's source code will be copied over to a Python file within `src/<package_name>/nodes/`:

```
kedro jupyter convert <filepath_to_my_notebook>
```

> _Note:_ The name of the Python file matches the name of the original notebook.

Alternatively, you may want to transform all your notebooks in one go. Run the following command to convert all notebook files found in the project root directory and under any of its sub-folders:

```
kedro jupyter convert --all
```

### How to ignore notebook output cells in `git`

To automatically strip out all output cell contents before committing to `git`, you can run `kedro activate-nbstripout`. This will add a hook in `.git/config` which will run `nbstripout` before anything is committed to `git`.

> _Note:_ Your output cells will be retained locally.

## Package your Kedro project

[Further information about building project documentation and packaging your project](https://kedro.readthedocs.io/en/stable/03_tutorial/05_package_a_project.html)
