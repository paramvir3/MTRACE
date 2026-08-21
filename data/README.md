# Training Data

Place an ASE-readable training trajectory at `data/train.extxyz`, or change
`train_file` in the selected YAML configuration. Each structure must contain a
total energy and Cartesian forces. Periodic structures should contain a cell
and periodic boundary flags. Stress is optional unless `stress_weight` is
nonzero; it may be stored as ASE-order stress or as an atomistic virial.

Trajectory data are intentionally not duplicated from the reference
Transformers-ACE repository. This keeps MTACE independent and prevents
large scientific datasets from being silently copied into a new repository.
