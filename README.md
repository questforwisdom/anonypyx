# AnonyPyx

This is a fork of the python library [AnonyPy](https://pypi.org/project/anonypy/) providing data anonymization techniques. 
AnonyPyx adds further algorithms (see below) and introduces a declarative interface.
If you consider migrating from AnonyPy, keep in mind that AnonyPyx is not compatible with its original API.

## Features

- partion-based anonymization algorithm Mondrian [1] supporting
    - k-anonymity
    - l-diversity 
    - t-closeness
- microclustering based anonymization algorithm MDAV-Generic [2] supporting
    - k-anonymity
- interoperability with pandas data frames
- supports both continuous and categorical attributes 
- image anonymization via the k-Same family of algorithms

## Install

```bash
pip install anonypyx
```


## Usage

**Disclaimer**: AnonyPyX does not shuffle the input data currently. In some applications, records can be re-identified based on the order in which they appear in the anonymized data set when shuffling is not used. 

Mondrian:

```python
import anonypyx
import pandas as pd

# Step 1: Prepare data as pandas data frame:

columns = ["age", "sex", "zip code", "diagnosis"]
data = [
    [50, "male", "02139", "stroke"],
    [33, "female", "10023", "flu"],
    [66, "intersex", "20001", "flu"],
    [28, "female", "33139", "diarrhea"],
    [92, "male", "94130", "cancer"],
    [19, "female", "96850", "diabetes"],
]

df = pd.DataFrame(data=data, columns=columns)

for column in ("sex", "zip code", "diagnosis"):
    df[column] = df[column].astype("category")

# Step 2: Prepare anonymizer

anonymizer = anonypyx.Anonymizer(df, k=3, l=2, algorithm="Mondrian", feature_columns=["age", "sex", "zip code"], sensitive_column="diagnosis")

# Step 3: Anonymize data (this might take a while for large data sets)

anonymized_records = anonymizer.anonymize()

# Print results:

anonymized_df = pd.DataFrame(anonymized_records)
print(anonymized_df)
```

Output: 

```bash
     age            sex           zip code diagnosis  count
0  19-33         female  10023,33139,96850  diabetes      1
1  19-33         female  10023,33139,96850  diarrhea      1
2  19-33         female  10023,33139,96850       flu      1
3  50-92  male,intersex  02139,20001,94130    cancer      1
4  50-92  male,intersex  02139,20001,94130       flu      1
5  50-92  male,intersex  02139,20001,94130    stroke      1
```

MDAV-generic:

```python
# Step 2: Prepare anonymizer
anonymizer = anonypyx.Anonymizer(df, k=3, algorithm="MDAV-generic", feature_columns=["age", "sex", "zip code"], sensitive_column="diagnosis")
```

k-Same-Eigen:

```python
import anonypyx
import numpy as np
import cv2

from os import listdir
from os.path import isfile, join

# Step 1: Load images into single numpy array

# images are loaded in grayscale
# every image must have the same height and width

path_to_dir = 'directory/containing/images/'
height = 120
width = 128
files = [f for f in listdir(path_to_dir) if isfile(join(path_to_dir, f))]
images = [cv2.imread(join(path_to_dir, f), flags = cv2.IMREAD_GRAYSCALE) for f in listdir(path_to_dir) if isfile(join(path_to_dir, f))]
images = np.array(images)

# Step 2: Prepare anonymizer

anonymizer = anonypyx.kSame(images, width, height, k=5, variant='eigen')

# Step 3: Anonymization

anonymized, mapping = anonymizer.anonymize()

# Display the first image and its anonymized version

sample_image = np.concatenate((images[0], anonymized[mapping[0]]), axis=1).astype('uint8')
sample_image = cv2.cvtColor(sample_image, cv2.COLOR_GRAY2BGR)
cv2.imshow("k-same-eigen", sample_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## Contributing

Clone the repository:

```bash
git clone https://github.com/questforwisdom/anonypyx.git
```

Set a virtual python environment up and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run tests:

```bash
pytest
```

## Changelog

### 0.2.0

- added the microaggregation algorithm MDAV-generic [2]
- added the Anonymizer class as the new API 
- removed Preserver class which was superseded by Anonymizer

### 0.2.1 - 0.2.3

- minor bugfixes

### 0.2.4

- added k-Same family of algorithms for image anonymization [3]
- added the microaggregation algorithm used by k-Same

## References

- [1]: LeFevre, K., DeWitt, D. J., & Ramakrishnan, R. (2006). Mondrian multidimensional K-anonymity. 22nd International Conference on Data Engineering (ICDE’06), 25–25. https://doi.org/10.1109/ICDE.2006.101
- [2]: Domingo-Ferrer, J., & Torra, V. (2005). Ordinal, continuous and heterogeneous k-anonymity through microaggregation. Data Mining and Knowledge Discovery, 11, 195–212.
- [3]: E. M. Newton, L. Sweeney, and B. Malin, ‘Preserving privacy by de-identifying face images’, IEEE Transactions on Knowledge and Data Engineering, vol. 17, no. 2, pp. 232–243, Feb. 2005, doi: 10.1109/TKDE.2005.32.

