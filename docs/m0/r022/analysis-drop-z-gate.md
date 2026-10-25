perturbation: drop-z-gate; cases: capital, long-context

## Hidden states per stage (max over cases, forwards and positions)

| stage | max abs | set by (case, forward, position, channel) | ref value there | stage max abs ref | max rel L2 (row) |
| --- | --- | --- | --- | --- | --- |
| embeddings | 0 | capital, 0, 0, 0 | 0.0143 | 0.14 | 0 |
| layer.0 | 3.13 | long-context, 0, 131, 0 | -0.322 | 0.993 | 9.12 |
| layer.1 | 3.34 | long-context, 0, 133, 0 | -0.404 | 1.05 | 8.57 |
| layer.2 | 4.75 | long-context, 18, 0, 0 | -0.754 | 1.6 | 9.55 |
| layer.3 | 4.37 | long-context, 0, 133, 0 | -0.278 | 1.33 | 8.11 |
| layer.4 | 5.09 | long-context, 0, 133, 0 | -0.367 | 1.31 | 8.62 |
| layer.5 | 5.28 | long-context, 0, 133, 0 | -0.156 | 1.38 | 10.1 |
| layer.6 | 5.14 | long-context, 0, 133, 0 | -0.172 | 4.15 | 10.9 |
| layer.7 | 5.13 | long-context, 0, 133, 0 | -0.353 | 2.18 | 10.4 |
| layer.8 | 5.31 | long-context, 0, 133, 0 | -0.343 | 2.15 | 11 |
| layer.9 | 5 | long-context, 0, 133, 0 | -0.409 | 2.1 | 11 |
| layer.10 | 4.66 | long-context, 18, 0, 0 | -0.594 | 2.27 | 11.3 |
| layer.11 | 4.81 | long-context, 18, 0, 0 | -0.47 | 1.16 | 10.3 |
| layer.12 | 5.19 | long-context, 18, 0, 0 | -0.47 | 1.27 | 10.4 |
| layer.13 | 5.47 | long-context, 0, 133, 0 | -0.783 | 1.6 | 9.75 |
| layer.14 | 5.57 | long-context, 0, 133, 0 | -0.693 | 4.64 | 8.27 |
| layer.15 | 5.61 | long-context, 18, 0, 0 | -0.798 | 1.95 | 6.73 |
| layer.16 | 8.37 | long-context, 0, 133, 0 | -0.996 | 2.48 | 6.81 |
| layer.17 | 9.87 | long-context, 0, 133, 0 | -1.06 | 2.83 | 6.57 |
| layer.18 | 13.3 | long-context, 0, 133, 0 | -3.09 | 4.98 | 5.95 |
| layer.19 | 12.6 | long-context, 0, 133, 0 | -2.33 | 4.86 | 5.72 |
| layer.20 | 14.4 | long-context, 0, 133, 0 | -1.93 | 5.81 | 5.37 |
| layer.21 | 15.8 | long-context, 0, 133, 0 | -2.39 | 6.24 | 4.74 |
| layer.22 | 17.5 | long-context, 0, 133, 0 | -3.39 | 6.44 | 4.09 |
| layer.23 | 15.6 | long-context, 0, 133, 0 | -2.27 | 8.4 | 3.9 |
| final_norm | 41.5 | capital, 0, 0, 123 | 39.3 | 39.9 | 1.64 |

## Hidden states per case (max over stages)

| case | prompt tokens | max abs, prefill | max abs, decode forwards | stage and forward of the max | max rel L2 |
| --- | --- | --- | --- | --- | --- |
| capital | 25 | 41.5 | 32 | final_norm f0 | 10.9 |
| long-context | 134 | 41.5 | 29.1 | final_norm f0 | 11.3 |

## Growth across forwards (max over cases; hidden over all stages, states over all linear layers)

| forward | hidden max abs | hidden, last layer (layer.23) | recurrent state max abs | conv state max abs |
| --- | --- | --- | --- | --- |
| 0 | 41.5 | 15.6 | 12.2 | 39.8 |
| 1 | 22.6 | 12.8 | 12.2 | 39.8 |
| 2 | 24.5 | 13.6 | 12.2 | 40.5 |
| 3 | 22.2 | 11.1 | 12.2 | 40.5 |
| 4 | 22.8 | 10.6 | 12.1 | 40.5 |
| 5 | 29.1 | 11.7 | 12.2 | 40.7 |
| 6 | 22.1 | 12.3 | 12.2 | 41.1 |
| 7 | 20.5 | 10.9 | 12.2 | 41.1 |
| 8 | 24.5 | 13 | 12.2 | 41.3 |
| 9 | 26.2 | 10.6 | 12.2 | 41.3 |
| 10 | 26.1 | 10.8 | 12.3 | 41.3 |
| 11 | 26.3 | 10.2 | 12.3 | 41.3 |
| 12 | 26.9 | 10.8 | 12.2 | 40.1 |
| 13 | 29.1 | 11.8 | 12.2 | 40.1 |
| 14 | 25.5 | 9.2 | 12.3 | 41.6 |
| 15 | 22.9 | 13.8 | 12.3 | 45.3 |
| 16 | 26.2 | 13.8 | 12.3 | 45.3 |
| 17 | 25 | 14 | 12.3 | 45.3 |
| 18 | 25.3 | 13.4 | 12.3 | 45.3 |
| 19 | 25.2 | 10.5 | 12.3 | 44.1 |
| 20 | 26.7 | 10.5 | 12.3 | 44.1 |
| 21 | 26.6 | 8.81 | 12.3 | 44.1 |
| 22 | 24.4 | 9.53 | 12.3 | 42.9 |
| 23 | 32 | 8.85 | 12.3 | 42.9 |

## recurrent_state per linear layer (max over cases and forwards)

| layer | max abs | set by (case, forward) | max abs ref | max rel L2 |
| --- | --- | --- | --- | --- |
| 0 | 0.0141 | long-context, 15 | 4.26 | 0.00366 |
| 1 | 1.84 | capital, 1 | 2.32 | 1 |
| 2 | 2.85 | capital, 5 | 2.16 | 1.27 |
| 4 | 12.3 | capital, 18 | 13.5 | 1 |
| 5 | 2.83 | capital, 18 | 2.87 | 1.26 |
| 6 | 1.93 | long-context, 0 | 1.49 | 1.36 |
| 8 | 4.42 | long-context, 4 | 4.47 | 1.03 |
| 9 | 2.82 | capital, 6 | 1.7 | 1.55 |
| 10 | 1.1 | long-context, 19 | 1.12 | 1.12 |
| 12 | 4.59 | capital, 22 | 4.8 | 1.12 |
| 13 | 1.28 | capital, 18 | 1.27 | 1.16 |
| 14 | 2.4 | long-context, 23 | 2.38 | 1.32 |
| 16 | 5.51 | capital, 19 | 5.14 | 1.22 |
| 17 | 3.76 | long-context, 1 | 3.9 | 1.21 |
| 18 | 8.17 | long-context, 22 | 8.18 | 1.1 |
| 20 | 5.6 | long-context, 17 | 5.64 | 1.38 |
| 21 | 3.5 | long-context, 22 | 3.51 | 1.41 |
| 22 | 2.37 | long-context, 22 | 2.38 | 1.25 |

## conv_state per linear layer (max over cases and forwards)

| layer | max abs | set by (case, forward) | max abs ref | max rel L2 | max abs, columns 1-3 (held by Roost) | max abs, column 0 (reconstructed) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.0607 | long-context, 0 | 28.3 | 0.00253 | 0.0607 | 0.0607 |
| 1 | 17 | long-context, 15 | 12.4 | 1.43 | 17 | 17 |
| 2 | 19.5 | long-context, 15 | 17.4 | 1.45 | 19.5 | 19.5 |
| 4 | 20.6 | long-context, 21 | 20.9 | 1.27 | 20.6 | 20.5 |
| 5 | 18 | capital, 18 | 15.8 | 1.23 | 18 | 18 |
| 6 | 16.8 | long-context, 16 | 15.8 | 1.31 | 16.8 | 16.8 |
| 8 | 22.7 | long-context, 2 | 18.8 | 1.23 | 22.7 | 22.7 |
| 9 | 17.9 | long-context, 13 | 16.3 | 1.35 | 17.9 | 17.9 |
| 10 | 17.1 | capital, 13 | 18.6 | 1.39 | 17.1 | 17.1 |
| 12 | 21.9 | long-context, 12 | 22.1 | 1.25 | 21.9 | 21.9 |
| 13 | 20.9 | capital, 12 | 15.8 | 1.36 | 20.9 | 20.9 |
| 14 | 22 | long-context, 10 | 20.4 | 1.35 | 22 | 22 |
| 16 | 32.6 | long-context, 0 | 28.4 | 1.3 | 32.6 | 32.6 |
| 17 | 44.1 | long-context, 18 | 34.7 | 1.31 | 44.1 | 44.1 |
| 18 | 34.7 | long-context, 16 | 28.8 | 1.24 | 34.7 | 34.7 |
| 20 | 26.8 | capital, 1 | 22.5 | 1.31 | 26.8 | 26.8 |
| 21 | 45.3 | long-context, 15 | 38.4 | 1.24 | 45.3 | 45.3 |
| 22 | 24.2 | long-context, 21 | 23.2 | 1.22 | 24.2 | 23.9 |

## Step-0 log-probabilities and per-forward argmax

| case | full-vector max abs | ref log-prob at that token | top-5 max abs | top-1 abs | step-0 argmax equal | reference step-0 margin | max abs margin error, all forwards | smallest reference margin (candidate's) | forwards whose argmax differs (reference margin) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| capital | 37.8 | -10.1 | 34.9 | 23.8 | False | 1.15 | 6.45 | 0.0554 (2.74) | f0 (1.15), f1 (0.797), f2 (1.71), f3 (3.11), f4 (1.26), f5 (0.722), f6 (2), f7 (0.535), f8 (2.08), f9 (1.04), f10 (4.04), f11 (0.547), f12 (1.09), f13 (0.725), f14 (2.77), f15 (0.0554), f16 (1.91), f17 (1.37), f18 (0.338), f19 (0.406), f20 (0.409), f21 (2.04), f22 (0.865), f23 (0.808) |
| long-context | 34.8 | -13.3 | 33.4 | 33.4 | False | 0.494 | 9.43 | 0.324 (4.12) | f0 (0.494), f1 (11.3), f2 (6.36), f3 (2.28), f4 (0.792), f5 (6.43), f6 (2.35), f7 (4.81), f8 (1.89), f9 (1.54), f10 (8.85), f11 (5.91), f12 (0.324), f13 (1.2), f14 (8.2), f15 (3.98), f16 (3.33), f17 (7.33), f18 (0.524), f19 (0.439), f20 (0.641), f21 (0.968), f22 (0.678), f23 (1.5) |

non-finite candidate hidden values: none
