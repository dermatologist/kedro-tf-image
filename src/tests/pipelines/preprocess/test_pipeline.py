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
This is a boilerplate test file for pipeline 'preprocess'
generated using Kedro 0.17.3.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
from pathlib import Path
from kedro.extras.datasets.pandas.csv_dataset import CSVDataSet
from kedro.framework.context.context import KedroContext
from kedro.pipeline import node
from kedro.pipeline.pipeline import Pipeline
import pytest
from kedro_tf_image.pipelines.preprocess.nodes import get_numeric_labels, get_tf_datasets, load_data_from_patitioned_dataset, load_data_from_url
from kedro.io import PartitionedDataSet
from tensorflow.keras.applications.resnet50 import preprocess_input


@pytest.fixture
def project_context():
    return KedroContext(
        package_name="kedro_tf_image",
        project_path=Path.cwd(),
    )


class TestPreprocesPipeline:
    # def test_csv_read(self, project_context):
    #     data_set = CSVDataSet(filepath="data/01_raw/skintype.csv")
    #     reloaded = data_set.load()
    #     print(load_data_from_url(reloaded))  # TODO Change this to assert

    # def test_image_read(self, project_context):
    #     dataset = {
    #         "type": "kedro_tf_image.extras.datasets.tf_image_dataset.TfImageDataSet",
    #         "preprocess_input": preprocess_input,
    #         "imagedim": 224
    #     }
    #     path= 'data/01_raw/imageset'
    #     filename_suffix= ".jpg"
    #     data_set = PartitionedDataSet(dataset=dataset, path=path, filename_suffix=filename_suffix)
    #     reloaded = data_set.load()
    #     data = load_data_from_patitioned_dataset(reloaded)  # TODO Change this to assert
    #     print(data['_cat_white_tan_16']['labels'])

    def test_tf_dataset(self, project_context):
        dataset = {
            "type": "kedro_tf_image.extras.datasets.tf_image_dataset.TfImageDataSet",
            "preprocess_input": preprocess_input,
            "imagedim": 224
        }
        path = 'data/01_raw/imageset'
        filename_suffix = ".jpg"
        data_set = PartitionedDataSet(
            dataset=dataset, path=path, filename_suffix=filename_suffix)
        reloaded = data_set.load()
        reloaded = load_data_from_patitioned_dataset(reloaded)
        data = get_tf_datasets(reloaded, params={'master_labels': ['cat', 'dog', 'white', 'black', 'tan']})  # TODO Change this to assert

    def test_get_labels(self, project_context):
        labels = ['cat', 'white']
        master_labels = ['cat', 'dog', 'white', 'black', 'tan']
        print(get_numeric_labels(labels, master_labels))
