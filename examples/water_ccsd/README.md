# Water Example

Place an ASE-readable water trajectory at `train.extxyz`, then run:

```bash
./run.sh
```

The default configuration learns energies and conservative forces. It does not
train stress because isolated-molecule trajectories normally do not carry a
physically meaningful periodic stress target.
