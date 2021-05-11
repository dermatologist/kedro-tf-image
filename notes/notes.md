dataset (Union[str, Type[AbstractDataSet], Dict[str, Any]]) – Underlying dataset definition. This is used to instantiate the dataset for each file located inside the path. Accepted formats are: a) object of a class that inherits from AbstractDataSet b) a string representing a fully qualified class name to such class c) a dictionary with type key pointing to a string from b), other keys are passed to the Dataset initializer. Credentials for the dataset can be explicitly specified in this configuration.

kedro ipython
skintypes = catalog.load("skintype_data")companies.head()
skintypes.head()

```
from kedro.extras.datasets.pandas import CSVDataSet
import pandas as pd

data = pd.DataFrame({'col1': [1, 2], 'col2': [4, 5],
                     'col3': [5, 6]})

# data_set = CSVDataSet(filepath="gcs://bucket/test.csv")
data_set = CSVDataSet(filepath="test.csv")
data_set.save(data)
reloaded = data_set.load()
assert data.equals(reloaded)
```