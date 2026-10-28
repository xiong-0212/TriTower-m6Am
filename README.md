# TriTower-m6Am

A triple-tower heterogeneous deep learning architecture for mRNA m6Am site prediction, integrating semantic (RNA-FM), sequential (One-Hot BiLSTM), and structural (RGCN) representations of RNA sequences.

## Overview

TriTower-m6Am combines three complementary towers through AUC-weighted ensemble learning:

- **RNA-FM Tower**: Pre-trained RNA language model embeddings with BellPooling position-aware aggregation
- **One-Hot BiLSTM Tower**: Bidirectional LSTM capturing sequential dependencies from one-hot encoded nucleotides
- **RGCN Tower**: Relational Graph Convolutional Network modeling RNA secondary structure with typed edges (backbone, base-pairing, neighborhood)

## Environment

| Component | Version |
|-----------|---------|
| Python | 3.9+ (tested on 3.13) |
| PyTorch | 2.0+ |
| CUDA | 11.8 / 12.1 |
| PyTorch Geometric | 2.4+ |
| scikit-learn | 1.3+ |
| NumPy | 1.24+ |

**Hardware:** Tested on a single NVIDIA RTX 3060 (12 GB) with 16 GB system RAM. Training on CPU is possible but significantly slower.

## Installation

```bash
pip install -r requirements.txt
```

Additional dependencies:
- [RNA-FM](https://github.com/ml4bio/RNA-FM) for embedding extraction (pretrained weights are auto-downloaded by the `fm` package)
- [ViennaRNA](https://www.tno.uni-vie.ac.at/cgi-bin/RNA/RNAfold.cgi) for secondary structure prediction

## Pretrained Checkpoints

Download the trained TriTower-m6Am model checkpoints (seed=123, 5-fold CV)
from the [v1.0.0 release](https://github.com/xiong-0212/TriTower-m6Am/releases/tag/v1.0.0):

```bash
# Download TriTower-m6Am_checkpoints.zip from the v1.0.0 release, then:
mkdir -p output
unzip TriTower-m6Am_checkpoints.zip -d output/
# Results in: output/ckpt_seed123/
```

Release asset SHA-256:

```text
F02A414315F519001DC30CA10FBBCB29FE14CAEB097C7EF73E5671E4DA863453
```

With checkpoints in place, training will automatically detect them and skip to evaluation (`-- SKIP (checkpoint found)`).

### Inference Using Pretrained Weights

```bash
python train.py
# Skips training, loads all 5-fold checkpoints, averages predictions,
# saves results to output/final_result.json
```

### RNA-FM Embedding Extraction

Pre-computed RNA-FM embeddings are not included due to file size constraints. Generate them before training:

```bash
mkdir -p data/rnafm
python -c "
from fm import pretrained
import numpy as np

model, alphabet = pretrained.rna_fm_t12()
batch_converter = alphabet.get_batch_converter()

def extract_embeddings(fasta_path, output_path):
    seqs = []
    with open(fasta_path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith('>'):
            seqs.append((lines[i].strip(), lines[i+1].strip()))
            i += 3
        else:
            i += 1
    batch_labels, batch_strs, batch_tokens = batch_converter(seqs)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[12])
    embeddings = results['representations'][12][:, 1:-1, :].numpy()
    np.save(output_path, embeddings.astype(np.float16))

extract_embeddings('data/train_ss_41.fasta', 'data/rnafm/rnafm_train_ss_41.npy')
extract_embeddings('data/test_ss_41.fasta', 'data/rnafm/rnafm_test_ss_41.npy')
"
```

## Repository Structure

```
TriTower-m6Am/
├── README.md
├── requirements.txt
├── train.py                     # Main training script
├── model.py                     # TriTower-m6Am model definitions
├── data/
│   ├── train_ss_41.fasta        # Training sequences with secondary structures
│   ├── test_ss_41.fasta         # Test sequences with secondary structures
│   ├── train_labels.npy         # Training labels
│   ├── test_labels.npy          # Test labels
│   └── rnafm/                   # Pre-computed RNA-FM embeddings
├── utils/
│   ├── preprocess.py            # Structural feature engineering & graph construction
│   ├── dataset.py               # Dataset classes & data loading
│   └── metrics.py               # Evaluation metrics
├── scripts/
│   ├── bca_pipeline/            # BCA negative sample construction & imbalanced evaluation
│   └── interpretability/        # IG attribution, SHAP, motif analysis, tower/edge contribution
├── configs/
│   └── default.yaml             # Hyperparameters and configuration
└── notebooks/
    └── figures.ipynb            # Figure generation notebook
```

## Usage

### Training

```bash
python train.py
```

This will train all three towers with 5-fold cross-validation and produce an AUC-weighted ensemble. Results are saved to `output/final_result.json`.

### Configuration

Edit `configs/default.yaml` to adjust hyperparameters, data paths, and training settings.

## Runtime Benchmark

All entries were measured on the same NVIDIA RTX 3060 (12 GB) with batch
size 1, using 20 warm-up passes followed by 200 timed passes (CUDA events).
Each component was benchmarked in its own process, and the reported times
are medians across three independent sessions. Parameters are the trainable
parameter count of each downstream encoder; the frozen RNA-FM backbone is
excluded.

| Model | Parameters | Inference time (ms) | Peak GPU memory (MB) |
|---|---|---|---|
| RNA-FM tower | 140,974 | 2.24 +/- 0.41 | 9.89 |
| One-Hot tower | 143,489 | 0.94 +/- 0.19 | 41.52 |
| RGCN tower | 45,203 | 8.61 +/- 0.68 | 9.33 |
| **TriTower ensemble** | **329,666** | **12.41 +/- 1.26** | **42.35** |
| **DTC-m6Am** | **1,446,256** | **3.6-8.5** | **15.63** |

The ensemble row covers only the three downstream encoders. It excludes
two per-window preprocessing steps required for deployment. RNAfold adds
2.81 ms per window, and a single RNA-FM embedding extraction at batch size 1
adds 18.33 ms and peaks at 391.46 MB. The complete single-window pipeline is
therefore approximately 33.6 ms with a peak of about 391 MB. DTC-m6Am was
bimodal across isolated runs (about 3.6 ms or 8.5 ms), so its latency is
reported as a range.

## Citation

If you use TriTower-m6Am in your research, please cite:

```bibtex
@article{xiong2026tritower,
  title={TriTower-m6Am: a triple-tower heterogeneous deep learning architecture
         integrating semantic, sequential, and structural information for
         mRNA N6,2'-O-dimethyladenosine site prediction},
  author={Xiong, Kaifeng and Jia, Jianhua},
  journal={PLOS Computational Biology},
  year={2026},
  note={In press}
}
```

## License

This project is released under the MIT License.
