perturbation: none; cases: capital, system-haiku, multi-turn, code, long-context

## Hidden states per stage (max over cases, forwards and positions)

| stage | max abs | set by (case, forward, position, channel) | ref value there | stage max abs ref | max rel L2 (row) |
| --- | --- | --- | --- | --- | --- |
| embeddings | 0 | capital, 0, 0, 0 | 0.0143 | 0.153 | 0 |
| layer.0 | 0.00591 | code, 19, 0, 0 | -0.584 | 0.999 | 0.0102 |
| layer.1 | 0.00928 | code, 19, 0, 0 | -0.685 | 1.05 | 0.0126 |
| layer.2 | 0.0159 | capital, 0, 14, 0 | -0.832 | 1.6 | 0.0155 |
| layer.3 | 0.0132 | system-haiku, 0, 25, 0 | -0.693 | 1.33 | 0.0147 |
| layer.4 | 0.0113 | system-haiku, 0, 25, 0 | -0.644 | 1.31 | 0.016 |
| layer.5 | 0.0149 | long-context, 0, 46, 0 | -0.749 | 1.38 | 0.0166 |
| layer.6 | 0.0364 | multi-turn, 0, 0, 119 | -4.15 | 4.15 | 0.0152 |
| layer.7 | 0.0515 | multi-turn, 0, 0, 119 | -2.18 | 2.18 | 0.0182 |
| layer.8 | 0.0362 | multi-turn, 0, 0, 119 | -2.15 | 2.15 | 0.0182 |
| layer.9 | 0.0278 | capital, 0, 0, 140 | -1.59 | 2.1 | 0.0185 |
| layer.10 | 0.0248 | capital, 0, 0, 140 | -1.71 | 2.27 | 0.0201 |
| layer.11 | 0.0222 | system-haiku, 0, 0, 140 | -0.779 | 1.16 | 0.0219 |
| layer.12 | 0.0224 | capital, 0, 0, 140 | -0.599 | 1.27 | 0.0255 |
| layer.13 | 0.0192 | system-haiku, 0, 0, 488 | -0.543 | 1.6 | 0.0236 |
| layer.14 | 0.0494 | capital, 0, 0, 488 | -4.01 | 4.64 | 0.0207 |
| layer.15 | 0.057 | capital, 0, 0, 140 | -1.07 | 1.95 | 0.0589 |
| layer.16 | 0.0555 | capital, 0, 0, 140 | -1.05 | 2.48 | 0.0576 |
| layer.17 | 0.057 | capital, 0, 0, 140 | -1.06 | 2.83 | 0.0566 |
| layer.18 | 0.0512 | system-haiku, 0, 0, 140 | -0.937 | 4.98 | 0.0462 |
| layer.19 | 0.05 | capital, 0, 0, 140 | -1.12 | 4.86 | 0.0516 |
| layer.20 | 0.0584 | system-haiku, 0, 5, 0 | -2.12 | 5.81 | 0.0566 |
| layer.21 | 0.0599 | long-context, 0, 56, 0 | -4.78 | 6.24 | 0.063 |
| layer.22 | 0.0706 | long-context, 0, 56, 0 | -4.46 | 6.44 | 0.058 |
| layer.23 | 0.138 | capital, 0, 21, 0 | -2.3 | 8.4 | 0.0515 |
| final_norm | 1.33 | multi-turn, 0, 33, 687 | 5.73 | 42.2 | 0.067 |

## Hidden states per case (max over stages)

| case | prompt tokens | max abs, prefill | max abs, decode forwards | stage and forward of the max | max rel L2 |
| --- | --- | --- | --- | --- | --- |
| capital | 25 | 0.761 | 0.286 | final_norm f0 | 0.067 |
| system-haiku | 44 | 1.1 | 0.293 | final_norm f0 | 0.067 |
| multi-turn | 69 | 1.33 | 0.407 | final_norm f0 | 0.067 |
| code | 31 | 0.761 | 0.342 | final_norm f0 | 0.067 |
| long-context | 134 | 0.78 | 0.311 | final_norm f0 | 0.067 |

## Growth across forwards (max over cases; hidden over all stages, states over all linear layers)

| forward | hidden max abs | hidden, last layer (layer.23) | recurrent state max abs | conv state max abs |
| --- | --- | --- | --- | --- |
| 0 | 1.33 | 0.138 | 0.0729 | 0.306 |
| 1 | 0.287 | 0.023 | 0.0715 | 0.306 |
| 2 | 0.267 | 0.0445 | 0.071 | 0.306 |
| 3 | 0.282 | 0.026 | 0.0717 | 0.306 |
| 4 | 0.234 | 0.0252 | 0.0693 | 0.166 |
| 5 | 0.357 | 0.0226 | 0.0692 | 0.17 |
| 6 | 0.407 | 0.0349 | 0.0689 | 0.17 |
| 7 | 0.242 | 0.0234 | 0.0691 | 0.17 |
| 8 | 0.256 | 0.0345 | 0.0684 | 0.17 |
| 9 | 0.243 | 0.0192 | 0.067 | 0.153 |
| 10 | 0.327 | 0.0524 | 0.0664 | 0.146 |
| 11 | 0.276 | 0.0179 | 0.0664 | 0.146 |
| 12 | 0.236 | 0.0209 | 0.0662 | 0.146 |
| 13 | 0.252 | 0.0587 | 0.0654 | 0.146 |
| 14 | 0.311 | 0.0852 | 0.0662 | 0.135 |
| 15 | 0.293 | 0.0369 | 0.0655 | 0.156 |
| 16 | 0.272 | 0.023 | 0.0653 | 0.156 |
| 17 | 0.259 | 0.0277 | 0.0665 | 0.156 |
| 18 | 0.227 | 0.019 | 0.0667 | 0.208 |
| 19 | 0.277 | 0.0286 | 0.0668 | 0.208 |
| 20 | 0.342 | 0.0464 | 0.0668 | 0.208 |
| 21 | 0.28 | 0.0201 | 0.0668 | 0.208 |
| 22 | 0.25 | 0.0323 | 0.0668 | 0.164 |
| 23 | 0.327 | 0.0391 | 0.0669 | 0.151 |

## recurrent_state per linear layer (max over cases and forwards)

| layer | max abs | set by (case, forward) | max abs ref | max rel L2 |
| --- | --- | --- | --- | --- |
| 0 | 0.015 | multi-turn, 19 | 4.62 | 0.00373 |
| 1 | 0.0166 | system-haiku, 20 | 2.36 | 0.00778 |
| 2 | 0.0149 | system-haiku, 17 | 2.28 | 0.00746 |
| 4 | 0.0338 | system-haiku, 5 | 13.9 | 0.00443 |
| 5 | 0.0521 | system-haiku, 6 | 2.95 | 0.00836 |
| 6 | 0.0117 | system-haiku, 11 | 1.49 | 0.00875 |
| 8 | 0.0202 | code, 10 | 4.61 | 0.00721 |
| 9 | 0.0231 | code, 17 | 2.61 | 0.00853 |
| 10 | 0.0111 | long-context, 18 | 1.36 | 0.0108 |
| 12 | 0.0391 | code, 0 | 4.88 | 0.00908 |
| 13 | 0.0162 | code, 23 | 1.27 | 0.00969 |
| 14 | 0.0167 | system-haiku, 0 | 2.56 | 0.00876 |
| 16 | 0.0729 | capital, 0 | 5.14 | 0.0162 |
| 17 | 0.0664 | multi-turn, 19 | 3.97 | 0.0148 |
| 18 | 0.0545 | capital, 0 | 8.18 | 0.0141 |
| 20 | 0.0514 | system-haiku, 1 | 5.75 | 0.0139 |
| 21 | 0.036 | system-haiku, 10 | 3.51 | 0.0165 |
| 22 | 0.0262 | code, 0 | 2.38 | 0.0154 |

## conv_state per linear layer (max over cases and forwards)

| layer | max abs | set by (case, forward) | max abs ref | max rel L2 | max abs, columns 1-3 (held by Roost) | max abs, column 0 (reconstructed) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.0607 | multi-turn, 0 | 32.3 | 0.00253 | 0.0607 | 0.0607 |
| 1 | 0.0671 | multi-turn, 10 | 14.8 | 0.00719 | 0.0671 | 0.0671 |
| 2 | 0.0827 | multi-turn, 14 | 21.9 | 0.00807 | 0.0827 | 0.0827 |
| 4 | 0.156 | multi-turn, 15 | 22.9 | 0.00727 | 0.156 | 0.156 |
| 5 | 0.11 | long-context, 2 | 19.1 | 0.00788 | 0.11 | 0.11 |
| 6 | 0.116 | code, 0 | 18 | 0.00905 | 0.116 | 0.116 |
| 8 | 0.121 | long-context, 0 | 18.8 | 0.00828 | 0.114 | 0.121 |
| 9 | 0.118 | code, 17 | 24.2 | 0.00994 | 0.118 | 0.118 |
| 10 | 0.132 | capital, 0 | 18.8 | 0.0113 | 0.132 | 0.132 |
| 12 | 0.145 | code, 0 | 22.3 | 0.01 | 0.145 | 0.145 |
| 13 | 0.156 | code, 0 | 16.6 | 0.0101 | 0.156 | 0.156 |
| 14 | 0.166 | code, 1 | 20.4 | 0.0106 | 0.166 | 0.166 |
| 16 | 0.144 | long-context, 21 | 28.5 | 0.008 | 0.144 | 0.141 |
| 17 | 0.183 | multi-turn, 18 | 34.7 | 0.00816 | 0.183 | 0.183 |
| 18 | 0.184 | long-context, 18 | 28.8 | 0.00777 | 0.184 | 0.184 |
| 20 | 0.165 | system-haiku, 0 | 22.8 | 0.00838 | 0.164 | 0.165 |
| 21 | 0.306 | code, 0 | 38.4 | 0.0098 | 0.306 | 0.306 |
| 22 | 0.175 | system-haiku, 0 | 23.2 | 0.00987 | 0.144 | 0.175 |

## Step-0 log-probabilities and per-forward argmax

| case | full-vector max abs | ref log-prob at that token | top-5 max abs | top-1 abs | step-0 argmax equal | reference step-0 margin | max abs margin error, all forwards | smallest reference margin (candidate's) | forwards whose argmax differs (reference margin) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| capital | 0.134 | -17.9 | 0.0299 | 0.0152 | True | 1.15 | 0.0797 | 0.0554 (0.0391) | none |
| system-haiku | 0.165 | -18.6 | 0.0625 | 0.0159 | True | 0.574 | 0.0808 | 0.0711 (0.0377) | none |
| multi-turn | 0.169 | -24.7 | 0.0305 | 0.0209 | True | 0.469 | 0.0989 | 0.0479 (0.0637) | none |
| code | 0.257 | -26.6 | 0.159 | 0.00151 | True | 4.1 | 0.115 | 0.00957 (0.0144) | none |
| long-context | 0.182 | -25.3 | 0.138 | 0.041 | True | 0.494 | 0.137 | 0.324 (0.289) | none |

non-finite candidate hidden values: none
