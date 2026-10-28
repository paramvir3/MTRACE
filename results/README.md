# Recorded experiment outputs

`cspbi3_matched_mixers.json` is the capacity-matched, three-seed mixer pilot
summarised in the manuscript's "What is established" section, produced by

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
accuracy claim. Read the manuscript's "What is established" and
"Limitations" sections before quoting a number from it.

`kernel_spectroscopy.json` is the untrained-prior mixer kernel, produced by
`python experiments/mixer_kernel_spectroscopy.py --json results/kernel_spectroscopy.json`.
It was regenerated when the script was corrected to measure the Jacobian
`d(y_k - u_k)/du_k'` itself; earlier versions measured `d||y_k||^2/du_k'`, which
weights each row by its own output and gave Mamba 0.783 L instead of 0.815 L.
