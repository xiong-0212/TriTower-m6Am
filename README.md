# TriTower-m6Am

A triple-tower heterogeneous deep learning architecture for m6Am site prediction, integrating semantic (RNA-FM), sequential (One-Hot BiLSTM), and structural (RGCN) representations of RNA sequences.

## Overview

TriTower-m6Am combines three complementary towers through AUC-weighted ensemble learning:

- **RNA-FM Tower**: Pre-trained RNA language model embeddings with BellPooling position-aware aggregation
- **One-Hot BiLSTM Tower**: Bidirectional LSTM capturing sequential dependencies from one-hot encoded nucleotides
- **RGCN Tower**: Relational Graph Convolutional Network modeling RNA secondary structure with typed edges (backbone, base-pairing, neighborhood)

## Installation

```bash
pip install -r requirements.txt
```

Additional dependencies:
- [RNA-FM](https://github.com/ml4bio/RNA-FM) for embedding extraction
- [ViennaRNA](https://www.tno.uni-vie.ac.at/cgi-bin/RNA/RNAfold.cgi) for secondary structure prediction

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

## Citation

If you use TriTower-m6Am in your research, please cite:

```bibtex
@article{xiong2025tritower,
  title={TriTower-m6Am: a triple-tower heterogeneous deep learning architecture
         integrating semantic, sequential, and structural information for
         N6,2'-O-dimethyladenosine site prediction},
  author={Xiong, Kaifeng and Jia, Jianhua},
  journal={},
  year={2025}
}
```

## License

This project is released under the MIT License.