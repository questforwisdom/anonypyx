# AnonyPyx

This is a fork of the python library [AnonyPy](https://pypi.org/project/anonypy/) providing data anonymisation techniques. 
AnonyPyx adds further algorithms (see below), introduces a declarative interface and adds attacks on the anonymised data.
If you consider migrating from AnonyPy, keep in mind that AnonyPyx is not compatible with its original API.

## Features

- partion-based anonymisation algorithm Mondrian [1] supporting
    - k-anonymity
    - l-diversity 
    - t-closeness
- microclustering based anonymisation algorithm MDAV-Generic [2] supporting
    - k-anonymity
- interoperability with pandas data frames
- supports both continuous and categorical attributes 
- image anonymisation via the k-Same family of algorithms [3]
- attacks on anonymised data sets:
    - intersection attack [4]
- privacy metrics:
    - percentage of vulnerable population [4]
- utility metrics:
    - error of aggregate queries [5]
    - discernibility penalty [6]

## Install

```bash
pip install anonypyx
```

## Usage

**Disclaimer**: AnonyPyX does not shuffle the input data currently. In some applications, records can be re-identified based on the order in which they appear in the anonymised data set when shuffling is not used. 

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

# Step 2: Prepare anonymiser

anonymiser = anonypyx.Anonymiser(
    df, k=3, l=2, algorithm="Mondrian", 
    feature_columns=["age", "sex", "zip code"], 
    sensitive_column="diagnosis",
    generalisation_strategy="human-readable"
)

# Step 3: Anonymise data (this might take a while for large data sets)

anonymised_records = anonymiser.anonymise()

# Print results:

anonymised_df = anonymised_records
print(anonymised_df)
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

- added k-Same family of algorithms for image anonymisation [3]
- added the microaggregation algorithm used by k-Same

### 0.2.5

- added the `generalisation_strategy` parameter to Anonymiser which determines how generalised data is represented (old behaviour is `"human-readable"`)
- added the generalisation strategies `"machine-readable"` and `"microaggregation"`
- added some attacks on anonymised data which can be found in the submodule `attackers`
- renamed `Anonymizer` and its `anonymize` method to BE spelling

### 0.2.6 - 0.2.8

- bugfixes

### 0.2.9

- added privacy metric *percentage of vulnerable population* [4]
- added counting query error as a utility metric [5]
- added the *discernibility penalty* [6] as a utility metric

## 0.2.10

- bugfixes

## 0.2.11

- added (de-)serialisation of generalisation schemas
- added `finalise()` method to all attackers for consistency
- removed dependency to `exact_multiset_cover`, anonypyx is platform-independent again
- renamed `prune_multiset_exact_cover()` method in `TrajectoryAttacker` to `finalise()`
- improved time efficiency of TrajectoryAttacker

## References

- [1]: LeFevre, K., DeWitt, D. J., & Ramakrishnan, R. (2006). Mondrian multidimensional K-anonymity. 22nd International Conference on Data Engineering (ICDE’06), 25–25. https://doi.org/10.1109/ICDE.2006.101
- [2]: Domingo-Ferrer, J., & Torra, V. (2005). Ordinal, continuous and heterogeneous k-anonymity through microaggregation. Data Mining and Knowledge Discovery, 11, 195–212.
- [3]: E. M. Newton, L. Sweeney, and B. Malin, ‘Preserving privacy by de-identifying face images’, IEEE Transactions on Knowledge and Data Engineering, vol. 17, no. 2, pp. 232–243, Feb. 2005, doi: 10.1109/TKDE.2005.32.
- [4]: Ganta, S. R., Kasiviswanathan, S. P., & Smith, A. (2008). Composition attacks and auxiliary information in data privacy. Proceedings of the 14th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 265–273.
- [5]: Xiao, X., & Tao, Y. (2007). M-invariance: Towards privacy preserving re-publication of dynamic datasets. Proceedings of the 2007 ACM SIGMOD International Conference on Management of Data, 689–700. https://doi.org/10.1145/1247480.1247556
- [6]: Bayardo, R. J., & Agrawal, R. (2005). Data privacy through optimal k-anonymization. 21st International Conference on Data Engineering (ICDE’05), 217–228. https://doi.org/10.1109/ICDE.2005.42



