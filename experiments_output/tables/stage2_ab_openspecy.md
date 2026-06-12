# Stage 2 Part B -- OpenSpecy A/B (RF)

CV is in-domain (held-out folds). FLOPP-e is the transfer signal. BLoP>=90% counts bioplastics forced into a commodity class with high confidence (lower is better; the model should be uncertain on OOD).

| Arm / config | n_train | pts | CV macro-F1 | PVC recall | HDPE rec | LDPE rec | FLOPP-e acc | BLoP forced >=90% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no_os__norm-snv | 6000 | 1868 | 0.999 | 1.000 | 1.000 | 1.000 | 96.4% (27/28) | 0/23 |
| no_os__asls+snv | 6000 | 1868 | 1.000 | 1.000 | 0.999 | 1.000 | 96.4% (27/28) | 0/23 |
| with_os__norm-snv | 14124 | 395 | 0.974 | 0.988 | 0.965 | 0.941 | 100.0% (28/28) | 0/23 |
| with_os__norm-snv_bal | 14124 | 395 | 0.975 | 0.986 | 0.969 | 0.943 | 100.0% (28/28) | 0/23 |
| with_os__asls+snv | 14124 | 395 | 0.979 | 0.997 | 0.974 | 0.948 | 92.9% (26/28) | 0/23 |
| with_os__asls+snv_bal | 14124 | 395 | 0.979 | 0.995 | 0.973 | 0.950 | 92.9% (26/28) | 0/23 |