# SpliceAI2

[Manuscript](https://assets.illumina.com/content/dam/illumina-marketing/images/genomics-research/articles/spliceai2/SpliceAI2.pdf) | [Blog post](https://www.illumina.com/science/genomics-research/articles/introducing-spliceai2--the-next-generation-of-splicing-and-trans.html)

This repository contains the source code for SpliceAI2, a deep learning model that predicts variant effects on splicing through quantitative modeling of splice sites, splice junctions, and transcripts.

Pretrained model weights and precomputed variant effect predictions within protein-coding genes (GENCODE v48, GRCh38), covering all possible single nucleotide variants (4 billion) and indels observed in human populations (150 million), are available on [Hugging Face](https://huggingface.co/illumina-ai) for academic and non-commercial research use. For commercial use, please contact [AI_licensing@illumina.com](mailto:AI_licensing@illumina.com). Precomputed scores are also available through [DRAGEN Annotation](https://help.connected.illumina.com/annotation).

SpliceAI2 predictions are summarized into a single `spliceai2_summary_score` ranging from 0 to 1, with higher values indicating larger predicted effects on splicing. Recommended thresholds, with their [SpliceAI](https://github.com/Illumina/SpliceAI) equivalents, are:

| SpliceAI2 | SpliceAI | Interpretation |
| - | - | - |
| 0.1 | 0.2 | High recall |
| 0.25 | 0.5 | Balance of precision and recall |
| 0.5 | 0.8 | High precision |

## Installation

SpliceAI2 is available through PyPI and can be installed with:

```bash
pip install spliceai2
```

Alternatively, clone the source repository and install SpliceAI2 in editable mode to work directly with the source code:

```bash
git clone https://github.com/Illumina/SpliceAI2.git
cd SpliceAI2
pip install -e .
```

A CUDA-capable GPU is required. If PyTorch raises a CUDA error, install a matching build from [pytorch.org](https://pytorch.org).

## Variant effect prediction

Provide variants in a `tsv` file with the following columns:

| Column | Description |
| - | - |
| `chrom` | Variant chromosome |
| `pos` | Variant position |
| `ref` | Reference allele; use nucleotide sequences for indels |
| `alt` | Alternate allele; use nucleotide sequences for indels |
| `strand` | Gene strand (`+` or `-`); if unknown, evaluate both strands separately and aggregate predictions |

Run SpliceAI2 with the pretrained model weights and a reference genome `fa` file:

```bash
spliceai2 \
    --model_folders /path/to/model_01 /path/to/model_02 \
    --var_tsv_file /path/to/variants.tsv \
    --fasta_file /path/to/reference.fa
```

Add `--no_compile` to skip `torch.compile` when scoring only a few variants or if it fails on your system.

The output file is saved alongside the input file, with `.spliceai2` appended to its name. Beyond the `spliceai2_summary_score`, the output contains 260 columns providing detailed splicing predictions organized as `{effect_type}_{field}_{k}`, as defined below.

| Effect type | Description |
| - | - |
| `donor_gain` | New or strengthened donor |
| `donor_loss` | Lost or weakened donor |
| `acceptor_gain` | New or strengthened acceptor |
| `acceptor_loss` | Lost or weakened acceptor |
| `jxn_gain` | New or strengthened junction |
| `jxn_loss` | Lost or weakened junction |

| Field | Description |
| - | - |
| `delta_score` | Absolute change in splicing probability |
| `ref_score` | Splicing probability for the reference allele |
| `alt_score` | Splicing probability for the alternate allele |
| `dist` | Distance between variant and splice site; for splice site effect types only |
| `donor_dist` | Distance between variant and donor site; for splice junction effect types only |
| `acceptor_dist` | Distance between variant and acceptor site; for splice junction effect types only |

`k` indexes the 10 strongest changes per effect type and ranges from 0 to 9, with 0 indicating the strongest change. The `spliceai2_summary_score` is the maximum of `donor_gain_delta_score_0`, `donor_loss_delta_score_0`, `acceptor_gain_delta_score_0`, and `acceptor_loss_delta_score_0`.

Eight of the 260 columns correspond one-to-one with SpliceAI's outputs: `DS_DG`, `DS_DL`, `DS_AG`, `DS_AL`, `DP_DG`, `DP_DL`, `DP_AG`, and `DP_AL` map to `donor_gain_delta_score_0`, `donor_loss_delta_score_0`, `acceptor_gain_delta_score_0`, `acceptor_loss_delta_score_0`, `donor_gain_dist_0`, `donor_loss_dist_0`, `acceptor_gain_dist_0`, and `acceptor_loss_dist_0`, respectively.

## Transcript prediction

SpliceAI2 can also predict transcripts directly from sequence. For variant analysis, generate predictions separately for the reference and alternate sequences.

```python
import torch
from spliceai2.model import SpliceAI2
from spliceai2.dataset import _one_hot_encode
from spliceai2.utils import decode_topk_tx

seq = "ACGT..."
# L + 131,072 nt on the sense strand, with L a multiple of 16
# Predictions span the central L positions

x = torch.tensor(_one_hot_encode(seq, "GRCh38")[None]).float().cuda()
models = [SpliceAI2.load_from_checkpoint(f"{f}/model.ckpt").eval().cuda()
          for f in ["/path/to/model_01", "/path/to/model_02"]]

with torch.no_grad():
    ftrs_list = [model.forward(x) for model in models]
    out_ss = torch.stack([
        model.forward_1d(ftrs) for model, ftrs in zip(models, ftrs_list)
    ]).mean(dim=0)
    out_1d_len = out_ss.shape[2]
    out_2d_len = int((out_ss.amax(dim=1).sigmoid() > 0.01).sum())
    out_2d_idxs = out_ss.amax(dim=1).topk(out_2d_len, dim=1)[1].sort(dim=1).values
    out_jxn = torch.stack([
        model.forward_2d(ftrs, out_2d_idxs) for model, ftrs in zip(models, ftrs_list)
    ]).mean(dim=0)
    out_tx = (
        out_ss[:, 0].gather(1, out_2d_idxs).unsqueeze(2)
        + out_jxn
        + out_ss[:, 1].gather(1, out_2d_idxs).unsqueeze(1)
    )

topk_tx = decode_topk_tx(
    out_tx[0].cpu().numpy(), out_2d_idxs[0].cpu().numpy(), out_1d_len, out_2d_len, k=3
)
# k × L array, one transcript per row; 1 is exonic and 0 is intronic
```

## Citation

Jaganathan K, Chen J, Liu X, Zhang Y, et al. A unified framework for quantitative splicing and transcript prediction. Unpublished manuscript (2026).

## Contact

Kishore Jaganathan: [kjaganathan@illumina.com](mailto:kjaganathan@illumina.com)
