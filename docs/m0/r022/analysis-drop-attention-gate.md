perturbation: drop-attention-gate; cases: capital, long-context

## Hidden states per stage (max over cases, forwards and positions)

| stage | max abs | set by (case, forward, position, channel) | ref value there | stage max abs ref | max rel L2 (row) |
| --- | --- | --- | --- | --- | --- |
| embeddings | 0 | capital, 0, 0, 0 | 0.0143 | 0.14 | 0 |
| layer.0 | 0.00505 | long-context, 0, 111, 0 | -0.534 | 0.993 | 0.00938 |
| layer.1 | 0.00911 | long-context, 0, 113, 0 | -0.876 | 1.05 | 0.0122 |
| layer.2 | 0.0159 | capital, 0, 14, 0 | -0.832 | 1.6 | 0.0155 |
| layer.3 | 5.87 | long-context, 0, 4, 0 | -0.405 | 1.33 | 5.82 |
| layer.4 | 5.78 | long-context, 0, 4, 0 | -0.625 | 1.31 | 4.95 |
| layer.5 | 5.68 | long-context, 0, 4, 0 | -0.678 | 1.38 | 4.4 |
| layer.6 | 5.7 | long-context, 0, 4, 0 | -0.733 | 4.15 | 4.1 |
| layer.7 | 6.76 | long-context, 0, 4, 0 | -0.76 | 2.18 | 5.29 |
| layer.8 | 6.94 | long-context, 0, 4, 0 | -0.685 | 2.15 | 5.33 |
| layer.9 | 6.81 | long-context, 0, 4, 0 | -0.752 | 2.1 | 5 |
| layer.10 | 6.73 | long-context, 0, 4, 0 | -0.609 | 2.27 | 4.82 |
| layer.11 | 6.76 | long-context, 0, 4, 0 | -0.694 | 1.16 | 4.99 |
| layer.12 | 6.72 | long-context, 0, 4, 0 | -0.684 | 1.27 | 4.68 |
| layer.13 | 6.74 | long-context, 0, 4, 0 | -0.865 | 1.6 | 4.27 |
| layer.14 | 6.32 | long-context, 0, 4, 0 | -0.63 | 4.64 | 3.34 |
| layer.15 | 7.2 | long-context, 0, 4, 0 | -1.1 | 1.95 | 3.48 |
| layer.16 | 7.59 | long-context, 0, 4, 0 | -1.53 | 2.48 | 2.97 |
| layer.17 | 7.43 | long-context, 0, 4, 0 | -1.81 | 2.83 | 2.54 |
| layer.18 | 8.78 | capital, 0, 17, 0 | -4.94 | 4.98 | 2.16 |
| layer.19 | 12.7 | long-context, 0, 16, 0 | -4.21 | 4.86 | 3.53 |
| layer.20 | 17.2 | capital, 0, 13, 0 | -3.84 | 5.81 | 3.56 |
| layer.21 | 17.1 | capital, 0, 12, 0 | -4.54 | 6.24 | 2.99 |
| layer.22 | 18.4 | long-context, 0, 49, 0 | -6.44 | 6.44 | 2.58 |
| layer.23 | 16.8 | long-context, 0, 16, 0 | -5.64 | 8.4 | 4.01 |
| final_norm | 58.5 | long-context, 0, 83, 3 | 30 | 39.9 | 1.53 |

## Hidden states per case (max over stages)

| case | prompt tokens | max abs, prefill | max abs, decode forwards | stage and forward of the max | max rel L2 |
| --- | --- | --- | --- | --- | --- |
| capital | 25 | 52.6 | 40.3 | final_norm f0 | 5.33 |
| long-context | 134 | 58.5 | 46.2 | final_norm f0 | 5.82 |

## Growth across forwards (max over cases; hidden over all stages, states over all linear layers)

| forward | hidden max abs | hidden, last layer (layer.23) | recurrent state max abs | conv state max abs |
| --- | --- | --- | --- | --- |
| 0 | 58.5 | 16.8 | 11.5 | 23.6 |
| 1 | 34.8 | 13.3 | 11.5 | 23.6 |
| 2 | 36.2 | 13.3 | 11.5 | 23.3 |
| 3 | 33.7 | 12.8 | 11.5 | 23.3 |
| 4 | 31.3 | 12.8 | 11.5 | 20.7 |
| 5 | 43.8 | 14 | 11.5 | 23.2 |
| 6 | 36.4 | 13.9 | 11.4 | 25.6 |
| 7 | 34.9 | 13.4 | 11.4 | 25.6 |
| 8 | 35.9 | 14.4 | 11.4 | 25.6 |
| 9 | 36 | 15 | 11.4 | 25.6 |
| 10 | 30.7 | 12.5 | 11.4 | 22.4 |
| 11 | 40.3 | 14.1 | 11.4 | 22.4 |
| 12 | 35.4 | 12.7 | 11.4 | 22.4 |
| 13 | 33.9 | 12.8 | 11.4 | 22.4 |
| 14 | 46.2 | 10.8 | 11.4 | 25.3 |
| 15 | 43.9 | 14.9 | 11.4 | 27 |
| 16 | 41.3 | 14.5 | 11.4 | 27 |
| 17 | 36.9 | 14.7 | 11.4 | 27 |
| 18 | 38.4 | 13.1 | 11.4 | 27 |
| 19 | 30.3 | 11.6 | 11.4 | 26.6 |
| 20 | 34.1 | 11.8 | 11.4 | 26.6 |
| 21 | 36.1 | 12.2 | 11.4 | 26.6 |
| 22 | 29.9 | 11.5 | 11.4 | 26.6 |
| 23 | 36.6 | 11.7 | 11.4 | 25.4 |

## recurrent_state per linear layer (max over cases and forwards)

| layer | max abs | set by (case, forward) | max abs ref | max rel L2 |
| --- | --- | --- | --- | --- |
| 0 | 0.0141 | long-context, 15 | 4.26 | 0.00366 |
| 1 | 0.0115 | long-context, 7 | 2.32 | 0.00687 |
| 2 | 0.0113 | capital, 19 | 2.16 | 0.00746 |
| 4 | 11.5 | capital, 1 | 13.5 | 0.891 |
| 5 | 2.5 | capital, 5 | 2.87 | 0.792 |
| 6 | 1.24 | capital, 19 | 1.49 | 0.89 |
| 8 | 4.29 | long-context, 4 | 4.47 | 0.958 |
| 9 | 1.5 | long-context, 18 | 1.7 | 0.989 |
| 10 | 1.1 | long-context, 19 | 1.12 | 0.972 |
| 12 | 4.06 | capital, 23 | 4.8 | 0.914 |
| 13 | 1.23 | capital, 18 | 1.27 | 0.899 |
| 14 | 2.36 | capital, 22 | 2.38 | 0.935 |
| 16 | 4.24 | long-context, 22 | 5.14 | 0.931 |
| 17 | 3.82 | long-context, 23 | 3.9 | 0.876 |
| 18 | 7.53 | long-context, 7 | 8.18 | 0.899 |
| 20 | 5.58 | long-context, 17 | 5.64 | 0.951 |
| 21 | 3.2 | long-context, 22 | 3.51 | 0.9 |
| 22 | 2.38 | long-context, 22 | 2.38 | 0.954 |

## conv_state per linear layer (max over cases and forwards)

| layer | max abs | set by (case, forward) | max abs ref | max rel L2 | max abs, columns 1-3 (held by Roost) | max abs, column 0 (reconstructed) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.0607 | long-context, 0 | 28.3 | 0.00253 | 0.0607 | 0.0607 |
| 1 | 0.0659 | capital, 18 | 12.4 | 0.0068 | 0.0659 | 0.0659 |
| 2 | 0.0776 | capital, 4 | 17.4 | 0.00722 | 0.0776 | 0.0776 |
| 4 | 15 | long-context, 16 | 20.9 | 0.786 | 15 | 15 |
| 5 | 17.8 | long-context, 2 | 15.8 | 0.82 | 17.8 | 17.8 |
| 6 | 11.3 | capital, 4 | 15.8 | 0.819 | 11.3 | 11.3 |
| 8 | 15.2 | long-context, 2 | 18.8 | 0.758 | 15.2 | 15.2 |
| 9 | 13.1 | long-context, 0 | 16.3 | 0.855 | 13.1 | 13.1 |
| 10 | 16.5 | capital, 0 | 18.6 | 0.893 | 16.5 | 16.5 |
| 12 | 15.1 | long-context, 5 | 22.1 | 0.785 | 15.1 | 15.1 |
| 13 | 13.7 | capital, 0 | 15.8 | 0.811 | 13.7 | 13.7 |
| 14 | 17.2 | capital, 2 | 20.4 | 0.855 | 17.2 | 17.2 |
| 16 | 17.9 | capital, 0 | 28.4 | 0.712 | 17.9 | 17.9 |
| 17 | 20.5 | capital, 0 | 34.7 | 0.695 | 20.5 | 20.5 |
| 18 | 23.6 | long-context, 0 | 28.8 | 0.614 | 23.6 | 23.6 |
| 20 | 18.9 | capital, 0 | 22.5 | 0.706 | 18.9 | 18.9 |
| 21 | 27 | long-context, 15 | 38.4 | 0.685 | 27 | 27 |
| 22 | 15.1 | long-context, 14 | 23.2 | 0.698 | 15.1 | 15.1 |

## Step-0 log-probabilities and per-forward argmax

| case | full-vector max abs | ref log-prob at that token | top-5 max abs | top-1 abs | step-0 argmax equal | reference step-0 margin | max abs margin error, all forwards | smallest reference margin (candidate's) | forwards whose argmax differs (reference margin) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| capital | 22.5 | -8.4 | 19.5 | 12.7 | False | 1.15 | 3.97 | 0.0554 (0.542) | f0 (1.15), f1 (0.797), f2 (1.71), f3 (3.11), f4 (1.26), f5 (0.722), f6 (2), f7 (0.535), f8 (2.08), f9 (1.04), f10 (4.04), f11 (0.547), f12 (1.09), f13 (0.725), f14 (2.77), f15 (0.0554), f16 (1.91), f17 (1.37), f18 (0.338), f19 (0.406), f20 (0.409), f21 (2.04), f22 (0.865), f23 (0.808) |
| long-context | 23.3 | -25.7 | 18.1 | 18.1 | False | 0.494 | 10.8 | 0.324 (0.199) | f0 (0.494), f1 (11.3), f2 (6.36), f3 (2.28), f4 (0.792), f5 (6.43), f6 (2.35), f7 (4.81), f8 (1.89), f9 (1.54), f10 (8.85), f11 (5.91), f12 (0.324), f13 (1.2), f14 (8.2), f16 (3.33), f17 (7.33), f18 (0.524), f19 (0.439), f20 (0.641), f21 (0.968), f22 (0.678), f23 (1.5) |

non-finite candidate hidden values: none
