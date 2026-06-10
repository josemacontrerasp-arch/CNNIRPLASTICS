# Stage 3 -- CNN vs RF head-to-head

CV = in-domain held-out folds. Lab/FLOPP-e = held-out generalization. BLoP>=90% = bioplastics forced into a commodity class with >=90% confidence (the CNN-vs-RF safety gap; lower is better).

| Run | Model | CV macro-F1 | Lab acc | FLOPP-e acc | BLoP forced >=90% |
|---|---|---:|---:|---:|---:|
| no_os__norm-snv | CNN | 0.998 | 92.3% (39) | 96.4% (27/28) | 14/23 |
| no_os__norm-snv | RF | 0.999 | 97.4% (39) | 96.4% (27/28) | 0/23 |
| no_os__smooth+d1+snv | CNN | 0.999 | 87.2% (39) | 96.4% (27/28) | 14/23 |
| no_os__smooth+d1+snv | RF | 1.000 | 92.3% (39) | 96.4% (27/28) | 0/23 |
| with_os__norm-snv | CNN | 0.971 | 89.7% (39) | 100.0% (28/28) | 14/23 |
| with_os__norm-snv | RF | 0.974 | 84.6% (39) | 100.0% (28/28) | 0/23 |