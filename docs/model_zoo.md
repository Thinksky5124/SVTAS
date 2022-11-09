# Model Zoo
## Model Zoo and Benchmarks

SVTAS provides reference implementation of a large number of video understanding approaches. In this document, we also provide comprehensive benchmarks to evaluate the supported models on different datasets using standard evaluation setup.

## TAS
### Gtea

model          | Params(M) | FLops(G) | modality           | F1@50% | F1@25% | F1@10% | Acc   | Edit  | Model
--------       | -----     | -------- | ------------------ | -----  | -----  | -----  | ----- | ----- | ------
MS-TCN         | 0.80      | 0.10     | RGB+Flow feature   | 74.60  | 85.40  | 87.50  | 79.20 | 81.40 | [link]()
Asformer       | 1.13      | 0.14     | RGB+Flow feature   | 79.80  | 87.80  | 79.70  | 97.7  | 84.60 | [link]()