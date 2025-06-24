# AnonyPyX User Guide

Welcome to AnonyPyX, a comprehensive toolkit designed for research on data anonymization. This guide provides an in-depth overview of the features and directs users to the relevant modules for further exploration. Detailed parameter information is available in the docstrings of classes and methods within each module.

## Overview
AnonyPyX is a *research* framework that offers a variety of anonymization techniques, attack simulations, generalization methods and evaluation metrics. AnonyPyX is designed **NOT** designed to sanitize data for real world applications.

## Key Features
- **Anonymization Algorithms**: Implements microaggregation, m-invariance and Mondrian algorithms to anonymize tabular data. Also implements the k-Same algorithms to sanitize face images.
- **Reconstruction Attacks**: Includes tools to simulate attacks such as the intersection attack to test the resilience of anonymized data.
- **Generalization Techniques**: Supports a variety of different represenations of generalized data.
- **Metrics and Evaluation**: Offers tools to measure the privacy and utility of anonymized data sets.

## Getting Started
To utilize AnonyPyX, explore the following modules based on your needs:

The `Anonymiser` class provides a declarative interface to the data anonymization pipeline.

## Submodule Contents

### Anonymization Algorithms

The anonymization process with AnonyPyX is split into two parts: (1) Creating a partitioning of the data which satisfies the selected privacy models and (2) generalising the partitions to create the actual sanitized data.
The anonymization algorithms implement the first step.
Privacy models are chosen when an instance of the algorithm's class is created.
All algorithms offer the method `partition(df)` to create the partitioning.

- **algorithms/**: Core anonymization algorithms.
    - `Mondrian`: Implementation of the multidimensional partition-based algorithm Mondrian. Supports *k*-anonymity, *l*-diversity and *t*-closeness. *(LeFevre, K., DeWitt, D. J., & Ramakrishnan, R. (2006). Mondrian multidimensional K-anonymity. 22nd International Conference on Data Engineering (ICDE’06), 25–25. https://doi.org/10.1109/ICDE.2006.101)*

  - `microaggregation.py`: Implements microaggregation for clustering and aggregating data.
  - `minvariance.py`: Applies minvariance techniques to balance privacy and utility.
  - `mondrian.py`: Utilizes Mondrian partitioning for multidimensional k-anonymity.

### Attackers
- **attackers/**: Simulate potential privacy breaches.
  - `base_attacker.py`: Provides a foundation for creating custom attack simulations.
  - `intersection_attacker.py`: Simulates attacks by intersecting datasets to re-identify individuals.
  - `trajectory_attacker.py`: Models attacks based on movement or sequence data.
  - `util.py`: Offers helper functions for attack implementation.

### Generalisation
- **generalisation/**: Data transformation and generalization methods.
  - `humanreadable.py`: Converts anonymized data into formats suitable for human interpretation.
  - `machinereadable.py`: Prepares data for automated processing by machines.
  - `microaggregation.py`: Applies generalization through clustering and aggregation.
  - `rawdata.py`: Handles initial data preprocessing for generalization.
  - `schema.py`: Defines structures for consistent data transformation.
  - `serialisation.py`: Manages the serialization of generalized data.

### Metrics
- **metrics/**: Evaluate anonymization outcomes.
  - `anonymiser.py`: Core class for evaluating anonymization processes.
  - `ksamme.py`: Implements k-anonymity and related metrics.
  - `models.py`: Provides supporting models for metric calculations.

## Next Steps
Begin by importing the desired module and class, e.g., `from algorithms import Microaggregation`. Refer to the docstrings for detailed parameter descriptions and usage examples. For advanced use cases, delve into the `attackers` module for testing robustness or the `generalisation` module for data transformation.

### Algorithms Guide
- **Microaggregation (`microaggregation.py`)**: Groups data into clusters and computes aggregate values (e.g., means) to reduce identifiability while preserving statistical properties.
- **Minvariance (`minvariance.py`)**: Optimizes the variance of anonymized data to ensure a balance between privacy protection and data utility.
- **Mondrian (`mondrian.py`)**: Applies a recursive partitioning approach to achieve k-anonymity across multiple dimensions.

### Attackers Guide
- **Base Attacker (`base_attacker.py`)**: Provides a foundation for creating custom attack simulations.
- **Intersection Attacker (`intersection_attacker.py`)**: Intersects datasets to re-identify individuals.
- **Utilities (`util.py`)**: Offers helper functions for attack implementation.

### Generalisation Guide
- **Human Readable (`humanreadable.py`)**: Converts anonymized data into formats suitable for human interpretation.
- **Machine Readable (`machinereadable.py`)**: Prepares data for automated processing by machines.
- **Microaggregation (`microaggregation.py`)**: Applies generalization through clustering and aggregation.
- **Raw Data (`rawdata.py`)**: Handles initial data preprocessing for generalization.
- **Schema (`schema.py`)**: Defines structures for consistent data transformation.
- **Serialisation (`serialisation.py`)**: Manages the serialization of generalized data.

### Metrics Guide
- **Anonymiser (`anonymiser.py`)**: Core class for evaluating anonymization processes.
- **K-Anonymity Metrics (`ksamme.py`)**: Measures compliance with k-anonymity standards.
- **Models (`models.py`)**: Provides supporting models for metric computations.
