perturbation: norm-eps-1e-5; cases: capital, long-context

## Hidden states per stage (max over cases, forwards and positions)

| stage | max abs | set by (case, forward, position, channel) | ref value there | stage max abs ref | max rel L2 (row) |
| --- | --- | --- | --- | --- | --- |
| embeddings | 0 | capital, 0, 0, 0 | 0.0143 | 0.14 | 0 |
| layer.0 | 0.453 | capital, 0, 21, 0 | -0.992 | 0.993 | 0.721 |
| layer.1 | 0.692 | long-context, 0, 86, 0 | -0.919 | 1.05 | 0.863 |
| layer.2 | 1.06 | capital, 0, 0, 0 | -1.6 | 1.6 | 0.909 |
| layer.3 | 0.878 | capital, 0, 1, 0 | -1.19 | 1.33 | 1 |
| layer.4 | 0.982 | capital, 0, 1, 0 | -1.11 | 1.31 | 0.96 |
| layer.5 | 0.868 | capital, 0, 1, 0 | -1.04 | 1.38 | 0.977 |
| layer.6 | 2.06 | capital, 0, 0, 119 | -4.15 | 4.15 | 0.995 |
| layer.7 | 1.8 | capital, 0, 0, 119 | -2.18 | 2.18 | 1.05 |
| layer.8 | 1.72 | capital, 0, 0, 119 | -2.15 | 2.15 | 1.05 |
| layer.9 | 1.67 | capital, 0, 0, 119 | -2.1 | 2.1 | 1.06 |
| layer.10 | 1.73 | capital, 0, 0, 119 | -2.27 | 2.27 | 1.06 |
| layer.11 | 0.907 | capital, 0, 0, 119 | -0.976 | 1.16 | 1.13 |
| layer.12 | 0.729 | capital, 0, 0, 119 | -0.815 | 1.27 | 1.11 |
| layer.13 | 0.766 | long-context, 0, 126, 0 | -0.846 | 1.6 | 1.08 |
| layer.14 | 3.96 | capital, 0, 0, 123 | 4.64 | 4.64 | 1.18 |
| layer.15 | 1.21 | capital, 0, 0, 123 | 1.55 | 1.95 | 1.1 |
| layer.16 | 1.34 | long-context, 0, 126, 0 | -1.4 | 2.48 | 1.1 |
| layer.17 | 1.55 | long-context, 0, 126, 0 | -1.61 | 2.83 | 1.09 |
| layer.18 | 2.66 | long-context, 0, 126, 0 | -4.34 | 4.98 | 0.985 |
| layer.19 | 2.13 | long-context, 0, 126, 0 | -3.03 | 4.86 | 1.09 |
| layer.20 | 2.46 | capital, 0, 0, 0 | 0.183 | 5.81 | 1.15 |
| layer.21 | 2.01 | capital, 23, 0, 0 | -1.44 | 6.24 | 1.14 |
| layer.22 | 3.22 | capital, 0, 0, 123 | 3.67 | 6.44 | 1.13 |
| layer.23 | 8.13 | capital, 0, 0, 123 | 8.4 | 8.4 | 1.28 |
| final_norm | 37.7 | capital, 0, 0, 123 | 39.3 | 39.9 | 1.41 |

## Hidden states per case (max over stages)

| case | prompt tokens | max abs, prefill | max abs, decode forwards | stage and forward of the max | max rel L2 |
| --- | --- | --- | --- | --- | --- |
| capital | 25 | 37.7 | 21.4 | final_norm f0 | 1.41 |
| long-context | 134 | 37.7 | 25.4 | final_norm f0 | 1.41 |

## Growth across forwards (max over cases; hidden over all stages, states over all linear layers)

| forward | hidden max abs | hidden, last layer (layer.23) | recurrent state max abs | conv state max abs |
| --- | --- | --- | --- | --- |
| 0 | 37.7 | 8.13 | 4.47 | 17 |
| 1 | 16.7 | 2.08 | 4.37 | 17 |
| 2 | 15.3 | 2 | 4.36 | 17 |
| 3 | 21.4 | 3.26 | 4.37 | 12.3 |
| 4 | 18.2 | 1.79 | 4.37 | 12.3 |
| 5 | 16.3 | 1.84 | 4.27 | 12.3 |
| 6 | 14.5 | 1.52 | 4.21 | 12 |
| 7 | 16.3 | 3.08 | 4.23 | 12 |
| 8 | 16.6 | 2.23 | 4.18 | 12 |
| 9 | 14 | 0.895 | 4.15 | 9.99 |
| 10 | 17.1 | 2.7 | 4.17 | 9.99 |
| 11 | 16.7 | 2.45 | 4.17 | 11.5 |
| 12 | 15.7 | 2.7 | 4.16 | 11.5 |
| 13 | 12.8 | 2.63 | 4.13 | 11.5 |
| 14 | 14.6 | 1.37 | 4.11 | 11.5 |
| 15 | 13.4 | 1.93 | 4.09 | 10.5 |
| 16 | 16.2 | 1.69 | 4.08 | 9.73 |
| 17 | 16.4 | 1.55 | 4.05 | 9.73 |
| 18 | 17.3 | 1.41 | 4.05 | 18.2 |
| 19 | 18.1 | 1.61 | 4.03 | 18.2 |
| 20 | 25.4 | 3.12 | 4.03 | 18.2 |
| 21 | 12.4 | 0.976 | 4.03 | 18.2 |
| 22 | 22.7 | 2.62 | 3.96 | 10.3 |
| 23 | 20.7 | 1.83 | 3.95 | 10.3 |

## recurrent_state per linear layer (max over cases and forwards)

| layer | max abs | set by (case, forward) | max abs ref | max rel L2 |
| --- | --- | --- | --- | --- |
| 0 | 0.0942 | long-context, 15 | 4.26 | 0.0207 |
| 1 | 1.17 | capital, 0 | 2.32 | 0.615 |
| 2 | 1.03 | capital, 3 | 2.16 | 0.557 |
| 4 | 2.18 | capital, 0 | 13.5 | 0.273 |
| 5 | 0.922 | long-context, 11 | 2.87 | 0.415 |
| 6 | 0.721 | long-context, 13 | 1.49 | 0.559 |
| 8 | 2.52 | capital, 0 | 4.47 | 0.601 |
| 9 | 0.511 | capital, 0 | 1.7 | 0.53 |
| 10 | 0.754 | capital, 1 | 1.12 | 0.691 |
| 12 | 2.58 | capital, 10 | 4.8 | 0.627 |
| 13 | 0.834 | long-context, 18 | 1.27 | 0.626 |
| 14 | 0.794 | long-context, 23 | 2.38 | 0.557 |
| 16 | 2.27 | capital, 2 | 5.14 | 0.741 |
| 17 | 1.96 | capital, 0 | 3.9 | 0.604 |
| 18 | 4.47 | capital, 0 | 8.18 | 0.641 |
| 20 | 2.79 | capital, 0 | 5.64 | 0.645 |
| 21 | 1.48 | capital, 8 | 3.51 | 0.656 |
| 22 | 1 | long-context, 0 | 2.38 | 0.611 |

## conv_state per linear layer (max over cases and forwards)

| layer | max abs | set by (case, forward) | max abs ref | max rel L2 | max abs, columns 1-3 (held by Roost) | max abs, column 0 (reconstructed) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.263 | capital, 10 | 28.3 | 0.0144 | 0.263 | 0.263 |
| 1 | 5.18 | long-context, 10 | 12.4 | 0.519 | 5.18 | 5.18 |
| 2 | 6.7 | long-context, 18 | 17.4 | 0.611 | 6.7 | 6.7 |
| 4 | 8.77 | long-context, 10 | 20.9 | 0.557 | 8.77 | 8.77 |
| 5 | 10.2 | long-context, 18 | 15.8 | 0.624 | 10.2 | 10.2 |
| 6 | 9.99 | long-context, 10 | 15.8 | 0.666 | 9.99 | 9.99 |
| 8 | 10.6 | long-context, 18 | 18.8 | 0.542 | 10.6 | 10.6 |
| 9 | 10.8 | long-context, 18 | 16.3 | 0.7 | 10.8 | 10.8 |
| 10 | 9.32 | long-context, 18 | 18.6 | 0.743 | 9.32 | 9.32 |
| 12 | 18.2 | long-context, 18 | 22.1 | 0.63 | 18.2 | 18.2 |
| 13 | 10.7 | long-context, 18 | 15.8 | 0.66 | 10.7 | 10.7 |
| 14 | 10.1 | long-context, 18 | 20.4 | 0.657 | 10.1 | 10.1 |
| 16 | 13.3 | long-context, 18 | 28.4 | 0.528 | 13.3 | 13.3 |
| 17 | 12.3 | long-context, 2 | 34.7 | 0.487 | 12.3 | 12.3 |
| 18 | 11.1 | capital, 0 | 28.8 | 0.452 | 11.1 | 11.1 |
| 20 | 17 | capital, 0 | 22.5 | 0.476 | 17 | 17 |
| 21 | 16.9 | long-context, 0 | 38.4 | 0.519 | 16.9 | 16.9 |
| 22 | 10.1 | capital, 0 | 23.2 | 0.527 | 10.1 | 10.1 |

## Step-0 log-probabilities and per-forward argmax

| case | full-vector max abs | ref log-prob at that token | top-5 max abs | top-1 abs | step-0 argmax equal | reference step-0 margin | max abs margin error, all forwards | smallest reference margin (candidate's) | forwards whose argmax differs (reference margin) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| capital | 15.2 | -27.2 | 7.41 | 2.9 | False | 1.15 | 1.78 | 0.0554 (0.0917) | f0 (1.15), f1 (0.797), f4 (1.26), f5 (0.722), f11 (0.547), f15 (0.0554), f16 (1.91), f17 (1.37), f18 (0.338), f20 (0.409), f21 (2.04), f23 (0.808) |
| long-context | 17 | -28.3 | 3.25 | 1.02 | False | 0.494 | 8.04 | 0.324 (1.63) | f0 (0.494), f3 (2.28), f4 (0.792), f6 (2.35), f7 (4.81), f12 (0.324), f14 (8.2), f19 (0.439), f20 (0.641), f21 (0.968), f22 (0.678), f23 (1.5) |

non-finite candidate hidden values: none
