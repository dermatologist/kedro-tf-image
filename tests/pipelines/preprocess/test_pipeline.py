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
from kedro.framework.context.context import KedroContext
from kedro.pipeline import node
from kedro.pipeline.pipeline import Pipeline
from kedro_tf_image.pipelines.preprocess import nodes as preprocess_nodes
import pandas as pd
import pytest


@pytest.fixture
def dummy_data():
    return pd.DataFrame(
        {
            "id": [1, 2],
            "url": [
                "https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/Sled_dog_on_Svalbard_with_heterochromia.jpg/640px-Sled_dog_on_Svalbard_with_heterochromia.jpg",
                "https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Dalmatian_fetching_a_stick.jpg/640px-Dalmatian_fetching_a_stick.jpg",
            ],
            "labels": ["cat|white|tan", "dog|black|white"],
        }
    )


@pytest.fixture
def dummy_parameters():
    parameters = {
        "model_options": {
            "test_size": 0.2,
            "random_state": 3,
            "features": ["engines", "passenger_capacity", "crew"],
        }
    }
    return parameters


def test_load_data_from_url(dummy_data, dummy_parameters):
    result = preprocess_nodes.load_data_from_url(dummy_data, delay=1, imagedim=224)
    assert isinstance(result, dict)
    assert len(result) == 2
