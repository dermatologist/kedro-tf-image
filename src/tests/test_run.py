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
This module contains an example test.

Tests should be placed in ``src/tests``, in modules that mirror your
project's structure, and in files named test_*.py. They are simply functions
named ``test_*`` which test a unit of logic.

To run the tests, run ``kedro test``.
"""
from pathlib import Path

import pytest
from kedro.framework.context import KedroContext

from kedro_tf_image.hooks import ProjectHooks
from kedro.extras.datasets.pandas import CSVDataSet
from kedro_tf_image.pipelines.preprocess.nodes import get_numeric_labels, get_tf_datasets, load_data_from_partitioned_dataset, load_data_from_url
from kedro.io import PartitionedDataSet
from tensorflow.keras.applications.resnet50 import preprocess_input
from kedro_tf_image.extras.datasets.tf_image_folder import TfImageFolder
from kedro_tf_image.extras.datasets.tf_image_generic import TfImageGeneric
from kedro_tf_image.extras.datasets.tf_image_processed import TfImageProcessed


@pytest.fixture
def project_context():
    return KedroContext(
        package_name="kedro_tf_image",
        project_path=Path.cwd()
    )


# The tests below are here for the demonstration purpose
# and should be replaced with the ones testing the project
# functionality
class TestProjectContext:

    # def test_csv_read(self, project_context):
    #     data_set = CSVDataSet(filepath="data/01_raw/skintype.csv")
    #     reloaded = data_set.load()
    #     print(load_data_from_url(reloaded))  # TODO Change this to assert

    def test_image_read(self, project_context):
        dataset = {
            "type": "kedro_tf_image.extras.datasets.tf_image_dataset.TfImageDataSet",
            "preprocess_input": "tensorflow.keras.applications.resnet50.preprocess_input",
            "imagedim": 224
        }
        path = 'data/01_raw/imageset'
        filename_suffix = ".jpg"
        data_set = PartitionedDataSet(
            dataset=dataset, path=path, filename_suffix=filename_suffix)
        reloaded = data_set.load()
        data = load_data_from_partitioned_dataset(
            reloaded)  # TODO Change this to assert
        assert data['_cat_white_tan_16']['labels'] == ['cat', 'white', 'tan']

    def test_tf_dataset(self, project_context):
        dataset = {
            "type": "kedro_tf_image.extras.datasets.tf_image_dataset.TfImageDataSet",
            "preprocess_input": "tensorflow.keras.applications.resnet50.preprocess_input",
            "imagedim": 224
        }
        path = 'data/01_raw/imageset'
        filename_suffix = ".jpg"
        data_set = PartitionedDataSet(
            dataset=dataset, path=path, filename_suffix=filename_suffix)
        reloaded = data_set.load()
        reloaded = load_data_from_partitioned_dataset(reloaded)
        (train_ds, val_ds) = get_tf_datasets(reloaded, params={'master_labels': [
            'cat', 'dog', 'white', 'black', 'tan'], 'val_size': 0.2})  # TODO Change this to assert
        for image, label in train_ds.take(1):
            assert image.numpy().shape == (1, 224, 224, 3)

    def test_get_labels(self, project_context):
        labels = ['cat', 'white']
        master_labels = ['cat', 'dog', 'white', 'black', 'tan']
        assert len(get_numeric_labels(labels, master_labels)) == 5

    def test_tf_folder(self, project_context):
        folderpath = "/home/a/archer/beapen/scratch/dermnet/train/rosacea-pd/tf"
        load_args = {
            "validation_split": 0.2,
            "seed": 123,
            "batch_size": 1
        }
        data_set = TfImageFolder(folderpath=folderpath, imagedim=224, load_args=load_args)
        (train_ds, val_ds) = data_set.load()
        for image, label in train_ds.take(1):
            assert image.numpy().shape == (1, 224, 224, 3)

    def test_tf_generic(self, project_context):
        filepath = "data/01_raw/imageset/_cat_black_white_15.jpg"
        load_args = {
            "target_size": (224, 224),
        }
        data_set = TfImageGeneric(filepath=filepath, imagedim=224, load_args=load_args)
        data = data_set.load()
        assert data is not None

    def test_load_dataset(self, project_context):
        folderpath = "data/02_intermediate/"
        data_set = TfImageProcessed(folderpath=folderpath, imagedim=224)
        data = data_set.load()
        assert data is not None
