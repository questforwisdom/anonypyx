# Overview of AnonyPyX

## Directory and Package Structure

- `anonypyx`: The main package.
    - `algorithms`: Algorithms partitioning raw data into equivalence classes satisfying specified syntactic privacy models.
    - `attackers`: Reconstruction attacks on generalised data frames.
    - `dlx`: Extension of Knuth's DLX algorithm to finite multisets. Used by some attacks.
    - `generalisation`: Generalisation strategies used to represent generalised data, for instance a human-readable strings or machine-readable intervals and sets.
    - `metrics`: Metrics for assessing the utility of generalised data frames and the success of attacks on them.
- `doc`: Documentation.
- `examples`: Some sample scripts demonstrating the use of AnonyPyX.
- `tests`: Unit tests of `anonypyx`.
