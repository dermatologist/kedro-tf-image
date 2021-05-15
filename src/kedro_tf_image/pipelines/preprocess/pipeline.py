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
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.17.3
"""

from kedro.pipeline import Pipeline, node

from kedro_tf_image.pipelines.preprocess.nodes import autotune, autotune_standardize, get_tf_datasets, load_data_from_patitioned_dataset, load_data_from_url, passon


# input = input and output = output
def create_download_pipeline(input="csvfilewithurls", output="imageset", **kwargs):
    return Pipeline([
                    node(
                        load_data_from_url, # Optional parameter delay, defaults to 3 seconds between each call
                        input,
                        output,
                        name="download_pipeline"
                    ),
    ])


def create_folder_pipeline(input="imagefolder", output="processeddataset", **kwargs):   # input = input and output = output
    return Pipeline([
                    node(
                        autotune_standardize,
                        input,
                        output,
                        name="folder_pipeline"
                    ),
                    ])


def create_passon_pipeline(input="imagefolder", output="datasetinmemory", **kwargs):
    return Pipeline([
                    node(
                        passon,
                        input,
                        output,
                        name="passon_pipeline"
                    ),
                    ])

# input = input and output = output
def create_multilabel_pipeline(input="imageset", output="processeddataset", **kwargs):
    return Pipeline([
                    node(
                        load_data_from_patitioned_dataset,
                        input,
                        "data_from_patitioned_dataset",
                        name="read_partitioned_data"
                    ),
                    node(
                        get_tf_datasets,
                        ["data_from_patitioned_dataset", "parameters"],
                        "datasetinmemory",  # requires copy_mode: assign
                        name="create_datasets"
                    ),
                    node(
                        autotune_standardize,
                        "datasetinmemory",
                        output,
                        name="multilabel_pipeline"
                    ),
                    ])
