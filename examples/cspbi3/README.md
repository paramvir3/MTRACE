# CsPbI3 Example

Place an ASE-readable trajectory at `train.extxyz`. Each frame must contain a
total energy, Cartesian forces, periodic cell, and stress or virial label.

Train architecture v8 from this directory:

```bash
python ../../train.py --config config.yaml
```

The run writes:

- `mtace_cspbi3.pt`: minimum validation-loss checkpoint;
- `mtace_cspbi3_last.pt`: last completed epoch.

Resume or warm-restart with:

```bash
python ../../train.py --config config.yaml --resume mtace_cspbi3_last.pt
python ../../train.py --config config.yaml --restart previous_model.pt
```

Export the best checkpoint:

```bash
mtace-export-lammps \
  mtace_cspbi3.pt \
  mtace_cspbi3_lammps.pt \
  --elements Cs Pb I
```

After building the included LAMMPS pair style, place a compatible
`cspbi3.data` file here and run:

```bash
"$LAMMPS_BUILD/lmp" -in in.mamba
```

Keep the LAMMPS type order `1=Cs`, `2=Pb`, `3=I`. Validate the potential on an
independent test set and with short NVE energy-drift runs before production MD.
