# AnonyPyX User Guide

Welcome to AnonyPyX, a comprehensive toolkit designed for data anonymization. This guide provides an in-depth overview of the features and directs users to the relevant modules for further exploration. Detailed parameter information is available in the docstrings of classes and methods within each module.

## Overview
AnonyPyX is a robust framework that offers a variety of anonymization techniques, attack simulations, generalization methods, and evaluation metrics. It is designed to help users protect sensitive data while preserving its utility for analysis.

## Key Features
- **Anonymization Algorithms**: Implements microaggregation, minvariance, and Mondrian techniques to anonymize data effectively.
- **Attack Simulations**: Includes tools to simulate attacks such as intersection and trajectory attacks to test the resilience of anonymized data.
- **Generalization Techniques**: Provides methods for transforming data into human-readable or machine-readable formats, including microaggregation and schema-based approaches.
- **Metrics and Evaluation**: Offers tools to measure the quality and effectiveness of anonymization processes.
- **Anonymizer Module**: A central component for applying various anonymization strategies with flexibility.

## Getting Started
To utilize AnonyPyX, explore the following modules based on your needs:

### Algorithms
- **algorithms/**: Core anonymization algorithms.
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
  - **Usage**: Import the specific algorithm module, e.g., `from algorithms import Mondrian`, and instantiate the class. Consult the docstrings for parameter details, such as input data formats and configuration options, along with example usage.

### Attackers Guide
- **Base Attacker (`base_attacker.py`)**: Provides a foundation for creating custom attack simulations.
- **Intersection Attacker (`intersection_attacker.py`)**: Simulates attacks by intersecting datasets to re-identify individuals.
- **Trajectory Attacker (`trajectory_attacker.py`)**: Models attacks based on movement or sequence data.
- **Utilities (`util.py`)**: Offers helper functions for attack implementation.
  - **Usage**: Import the relevant module, e.g., `from attackers import IntersectionAttacker`, and review the docstrings for setup instructions and parameter options to configure attack scenarios.

### Generalisation Guide
- **Human Readable (`humanreadable.py`)**: Converts anonymized data into formats suitable for human interpretation.
- **Machine Readable (`machinereadable.py`)**: Prepares data for automated processing by machines.
- **Microaggregation (`microaggregation.py`)**: Applies generalization through clustering and aggregation.
- **Raw Data (`rawdata.py`)**: Handles initial data preprocessing for generalization.
- **Schema (`schema.py`)**: Defines structures for consistent data transformation.
- **Serialisation (`serialisation.py`)**: Manages the serialization of generalized data.
  - **Usage**: Import the desired module, e.g., `from generalisation import HumanReadable`, and check the docstrings for parameter details and examples of data transformation workflows.

### Metrics Guide
- **Anonymiser (`anonymiser.py`)**: Core class for evaluating anonymization processes.
- **K-Anonymity Metrics (`ksamme.py`)**: Measures compliance with k-anonymity standards.
- **Models (`models.py`)**: Provides supporting models for metric computations.
  - **Usage**: Import the module, e.g., `from metrics import Anonymiser`, and explore the docstrings for parameter configurations and examples of metric application.