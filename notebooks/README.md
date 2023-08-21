# `notebooks/` Structure

- 1-data-preparation
    - convert the original data to csv to improve data loading speed.
- 2-data-exploration
    - 1-detect-place-cells
        - neural analysis
        - mutual information, firing rate
    - 2-check-movement
        - behavioral analysis
        - velocity, stay period in one position.
    - 3-effect-of-discretization
        - firing rate and velocity at each position.
- 3-decoding-model
    - 1-varied-num-of-samples
        - implementation of ROCKET based on varied number of samples from each position label.
    - 2-uniform-num-of-samples
        - implementation of ROCKET based on uniform number of samples from each position label.
- 4-individual-analysis
    - difference between ns and sig mice.