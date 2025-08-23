"""Project pipelines."""

from typing import Dict

from kedro.pipeline import Pipeline


from kedro_tf_image.pipelines import preprocess as preprocess


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """

    download = preprocess.create_download_pipeline(
        input="skintype_data", output="imageset"
    )  # input is csv
    folder = preprocess.create_folder_pipeline(
        input="imagefolder", output="processeddataset"
    )
    multilabel = preprocess.create_multilabel_pipeline(
        input="imageset", output="processeddataset"
    )

    return {
        "download": download,
        "multilabel": multilabel,
        "folder": folder,
        "__default__": folder,
    }
