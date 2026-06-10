# Stage 4 -- reject-option threshold (headline model)

| Model | AUROC known/OOD | Youden t | t@95% OOD-reject | known kept @t | acc of kept | OOD forced >=0.9 |
|---|---:|---:|---:|---:|---:|---:|
| CNN | 0.836 | 1.000 | inf | 0% | nan% | 34/57 |
| RF | 0.997 | 0.900 | 0.890 | 98% | 100.0% | 1/57 |

## Alternative: explicit OTHER class (RF, OpenSpecy bioplastics)

- CV macro-F1 (7-class): 0.965
- BLoP routed to OTHER: 0/23 (0%)
- NIST probes routed to OTHER: 0/3
- frac_other = share of external OOD correctly sent to OTHER; compare to the threshold reject rate in models.rf.ood_rejected_at_t95