# Recorded experiment outputs

`cspbi3_matched_mixers.json` is the capacity-matched, three-seed mixer pilot
described in `docs/PILOT_RESULTS.md`, produced by

```bash
python experiments/matched_mixer_study.py \
  --train examples/cspbi3/train.extxyz --frames 240 --stride 4 \
  --epochs 30 --seeds 3 --match-parameters --mimo-rank 1 \
  --num-shells 16 --shell-r-min 2.0 \
  --output results/cspbi3_matched_mixers.json
```

It contains the full configuration, the resolved per-mixer widths, per-epoch
validation histories for every seed, and the gate-dependence diagnostic. It is a
pilot: 40 training frames, 30 epochs, one composition. It does not support any
accuracy claim. Read the limitations section of `docs/PILOT_RESULTS.md` before
quoting a number from it.
