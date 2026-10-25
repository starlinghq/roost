definition defect (PagedAttention WindowLeft unbound, so 0), no perturbation; cases: capital, system-haiku, multi-turn, code, long-context

## Hidden states per stage (max over cases, forwards and positions)

| stage | max abs | set by (case, forward, position, channel) | ref value there | stage max abs ref | max rel L2 (row) |
| --- | --- | --- | --- | --- | --- |
| embeddings | 0 | capital, 0, 0, 0 | 0.0143 | 0.153 | 0 |
| layer.0 | 0.00591 | code, 19, 0, 0 | -0.584 | 0.999 | 0.0102 |
| layer.1 | 0.00928 | code, 19, 0, 0 | -0.685 | 1.05 | 0.0126 |
| layer.2 | 0.0159 | capital, 0, 14, 0 | -0.832 | 1.6 | 0.0155 |
| layer.3 | 0.61 | long-context, 0, 121, 0 | -0.664 | 1.33 | 0.899 |
| layer.4 | 0.308 | system-haiku, 0, 39, 0 | -0.656 | 1.31 | 0.697 |
| layer.5 | 0.281 | long-context, 0, 120, 0 | -0.692 | 1.38 | 0.631 |
| layer.6 | 0.211 | multi-turn, 0, 34, 0 | -0.375 | 4.15 | 0.599 |
| layer.7 | 0.513 | long-context, 0, 120, 0 | -0.659 | 2.18 | 0.866 |
| layer.8 | 0.529 | long-context, 0, 92, 0 | -0.35 | 2.15 | 0.892 |
| layer.9 | 0.462 | long-context, 0, 83, 0 | -0.756 | 2.1 | 0.839 |
| layer.10 | 0.465 | long-context, 0, 48, 0 | -1.19 | 2.27 | 0.831 |
| layer.11 | 0.836 | system-haiku, 19, 0, 0 | -0.375 | 1.16 | 1.08 |
| layer.12 | 0.793 | long-context, 17, 0, 0 | -0.164 | 1.27 | 1.03 |
| layer.13 | 0.891 | long-context, 4, 0, 0 | -0.453 | 1.6 | 1 |
| layer.14 | 1.17 | long-context, 21, 0, 0 | -0.694 | 4.64 | 1.01 |
| layer.15 | 1.23 | multi-turn, 0, 27, 0 | -0.593 | 1.95 | 1.08 |
| layer.16 | 1.48 | long-context, 18, 0, 0 | -0.645 | 2.48 | 1.08 |
| layer.17 | 1.83 | long-context, 18, 0, 0 | -0.608 | 2.83 | 1.08 |
| layer.18 | 2.55 | long-context, 4, 0, 0 | -0.12 | 4.98 | 1.05 |
| layer.19 | 2.63 | long-context, 18, 0, 0 | -1.87 | 4.86 | 1.01 |
| layer.20 | 2.23 | multi-turn, 0, 27, 0 | -2.33 | 5.81 | 0.966 |
| layer.21 | 2.4 | long-context, 23, 0, 0 | -2.16 | 6.24 | 0.944 |
| layer.22 | 2.58 | system-haiku, 0, 43, 0 | -2.33 | 6.44 | 0.952 |
| layer.23 | 5.83 | long-context, 14, 0, 0 | 0.0849 | 8.4 | 1.62 |
| final_norm | 27.1 | multi-turn, 0, 33, 713 | -18.5 | 42.2 | 1.12 |

## Hidden states per case (max over stages)

| case | prompt tokens | max abs, prefill | max abs, decode forwards | stage and forward of the max | max rel L2 |
| --- | --- | --- | --- | --- | --- |
| capital | 25 | 22 | 16.9 | final_norm f0 | 1.02 |
| system-haiku | 44 | 23.6 | 18.4 | final_norm f0 | 1.12 |
| multi-turn | 69 | 27.1 | 24.4 | final_norm f0 | 1.12 |
| code | 31 | 13.4 | 18.7 | final_norm f6 | 0.993 |
| long-context | 134 | 17.2 | 22.2 | final_norm f22 | 1.62 |

## Growth across forwards (max over cases; hidden over all stages, states over all linear layers)

| forward | hidden max abs | hidden, last layer (layer.23) | recurrent state max abs | conv state max abs |
| --- | --- | --- | --- | --- |
| 0 | 27.1 | 4.1 | 2.16 | 11.4 |
| 1 | 24.4 | 3.79 | 2.18 | 11.4 |
| 2 | 21.2 | 3.6 | 2.16 | 11.4 |
| 3 | 23.3 | 2.37 | 2 | 11.4 |
| 4 | 18.3 | 2.71 | 1.51 | 10.2 |
| 5 | 17.3 | 4.6 | 1.59 | 10.2 |
| 6 | 20.9 | 3.34 | 1.5 | 10.2 |
| 7 | 17.2 | 2.79 | 1.54 | 10.2 |
| 8 | 21 | 2.49 | 1.53 | 9.66 |
| 9 | 21.2 | 2.29 | 1.47 | 9.66 |
| 10 | 21 | 2.47 | 1.54 | 9.66 |
| 11 | 17.5 | 3.15 | 1.45 | 15.8 |
| 12 | 15.4 | 2.21 | 1.65 | 15.8 |
| 13 | 21.2 | 3.38 | 1.44 | 15.8 |
| 14 | 22 | 5.83 | 1.49 | 15.8 |
| 15 | 18.2 | 2.26 | 1.4 | 11.7 |
| 16 | 16.9 | 2.77 | 1.51 | 11.7 |
| 17 | 21.6 | 2.77 | 1.51 | 11.7 |
| 18 | 13.7 | 2.14 | 1.52 | 11.4 |
| 19 | 15.9 | 3.72 | 1.62 | 14.3 |
| 20 | 15.1 | 2.16 | 1.7 | 14.3 |
| 21 | 14.9 | 4.03 | 1.82 | 14.3 |
| 22 | 22.2 | 4.43 | 1.8 | 14.3 |
| 23 | 15.7 | 2.98 | 1.9 | 11 |

## recurrent_state per linear layer (max over cases and forwards)

| layer | max abs | set by (case, forward) | max abs ref | max rel L2 |
| --- | --- | --- | --- | --- |
| 0 | 0.015 | multi-turn, 19 | 4.62 | 0.00373 |
| 1 | 0.0166 | system-haiku, 20 | 2.36 | 0.00778 |
| 2 | 0.0149 | system-haiku, 17 | 2.28 | 0.00746 |
| 4 | 1.22 | long-context, 21 | 13.9 | 0.197 |
| 5 | 0.539 | system-haiku, 20 | 2.95 | 0.261 |
| 6 | 0.449 | multi-turn, 7 | 1.49 | 0.372 |
| 8 | 0.898 | long-context, 23 | 4.61 | 0.492 |
| 9 | 0.727 | multi-turn, 11 | 2.61 | 0.566 |
| 10 | 0.687 | system-haiku, 22 | 1.36 | 0.688 |
| 12 | 1.03 | system-haiku, 21 | 4.88 | 0.521 |
| 13 | 1.1 | long-context, 23 | 1.27 | 0.674 |
| 14 | 0.978 | system-haiku, 22 | 2.56 | 0.632 |
| 16 | 2.18 | long-context, 1 | 5.14 | 0.455 |
| 17 | 1.16 | long-context, 0 | 3.97 | 0.425 |
| 18 | 1.83 | long-context, 0 | 8.18 | 0.327 |
| 20 | 1.13 | multi-turn, 2 | 5.75 | 0.327 |
| 21 | 1.52 | system-haiku, 22 | 3.51 | 0.48 |
| 22 | 0.738 | code, 0 | 2.38 | 0.48 |

## conv_state per linear layer (max over cases and forwards)

| layer | max abs | set by (case, forward) | max abs ref | max rel L2 | max abs, columns 1-3 (held by Roost) | max abs, column 0 (reconstructed) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.0607 | multi-turn, 0 | 32.3 | 0.00253 | 0.0607 | 0.0607 |
| 1 | 0.0671 | multi-turn, 10 | 14.8 | 0.00719 | 0.0671 | 0.0671 |
| 2 | 0.0827 | multi-turn, 14 | 21.9 | 0.00807 | 0.0827 | 0.0827 |
| 4 | 7.62 | multi-turn, 5 | 22.9 | 0.395 | 7.62 | 7.62 |
| 5 | 6.57 | multi-turn, 6 | 19.1 | 0.4 | 6.57 | 6.57 |
| 6 | 5.68 | multi-turn, 6 | 18 | 0.409 | 5.68 | 5.68 |
| 8 | 7.57 | long-context, 23 | 18.8 | 0.465 | 7.57 | 6.93 |
| 9 | 8.94 | long-context, 23 | 24.2 | 0.597 | 8.94 | 8.05 |
| 10 | 6.87 | system-haiku, 19 | 18.8 | 0.688 | 6.87 | 6.87 |
| 12 | 9.65 | capital, 0 | 22.3 | 0.632 | 9.65 | 9.65 |
| 13 | 9.83 | long-context, 21 | 16.6 | 0.693 | 9.83 | 9.59 |
| 14 | 8.7 | system-haiku, 0 | 20.4 | 0.677 | 8.7 | 8.7 |
| 16 | 10.5 | system-haiku, 0 | 28.5 | 0.481 | 10.5 | 10.5 |
| 17 | 15.8 | multi-turn, 11 | 34.7 | 0.441 | 15.8 | 15.8 |
| 18 | 10.4 | system-haiku, 0 | 28.8 | 0.375 | 10.4 | 10.4 |
| 20 | 14.3 | system-haiku, 19 | 22.8 | 0.387 | 14.3 | 14.3 |
| 21 | 10.8 | system-haiku, 0 | 38.4 | 0.424 | 10.8 | 10.8 |
| 22 | 9.63 | multi-turn, 0 | 23.2 | 0.435 | 9.63 | 9.63 |

## Step-0 log-probabilities and per-forward argmax

| case | full-vector max abs | top-5 max abs | top-1 abs | step-0 argmax equal | reference step-0 margin | forwards whose argmax differs (reference margin) |
| --- | --- | --- | --- | --- | --- | --- |
| capital | 9.51 | 3.22 | 0.584 | True | 1.15 | f1 (0.797), f4 (1.26), f5 (0.722), f15 (0.0554), f17 (1.37), f18 (0.338), f23 (0.808) |
| system-haiku | 11.6 | 8.19 | 8.19 | False | 0.574 | f0 (0.574), f1 (1.15), f4 (1.1), f6 (2.16), f7 (0.126), f8 (1.15), f12 (0.273), f14 (0.983), f16 (0.892), f17 (1.13), f18 (2.26), f21 (2.48), f22 (3.77) |
| multi-turn | 15.3 | 3.52 | 1.31 | False | 0.469 | f0 (0.469), f9 (0.0479), f18 (0.887), f19 (0.981), f20 (0.247), f23 (0.909) |
| code | 12.5 | 5.31 | 2.47 | False | 4.1 | f0 (4.1), f17 (0.19), f19 (2.94), f21 (6.5), f23 (0.00957) |
| long-context | 14.6 | 8.18 | 0.917 | False | 0.494 | f0 (0.494), f3 (2.28), f4 (0.792), f6 (2.35), f7 (4.81), f8 (1.89), f9 (1.54), f10 (8.85), f11 (5.91), f12 (0.324), f13 (1.2), f16 (3.33), f19 (0.439), f20 (0.641), f21 (0.968), f22 (0.678), f23 (1.5) |

non-finite candidate hidden values: none
