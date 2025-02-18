## Deep Cross Attention (wip)

Implementation of the proposed [DeepCrossAttention](https://arxiv.org/abs/2502.06785) by Heddes et al while at Google research, in Pytorch

My analysis is although I still prefer [Hyper Connections](https://arxiv.org/abs/2409.19606), they have an important idea here that I have been trying concurrently. Mainly the queries, keys, values can be [routed from different layers](https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py#L1226) of the past

## Citations

```bibtex
@inproceedings{Heddes2025DeepCrossAttentionST,
    title   = {DeepCrossAttention: Supercharging Transformer Residual Connections},
    author  = {Mike Heddes and Adel Javanmard and Kyriakos Axiotis and Gang Fu and MohammadHossein Bateni and Vahab S. Mirrokni},
    year    = {2025},
    url     = {https://api.semanticscholar.org/CorpusID:276250576}
}
```
