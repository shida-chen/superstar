# Dynamic Pruning

This repositories is used for deep learning models under dynamic pruning training, which aims to accelerate the training process by reducing the computations in the training procedure.

## Workflow

| 1st phase | 2rd phase |3rd phase|
|  ----  | ----  | ---- |
|**Pre-training** |**Pruning** |**Recovering**|   

where the `1st` is used for making models be able to inference, `2nd` is to reduce the computations and decrease the complexity, while `3rd` is to offset the training loss.

