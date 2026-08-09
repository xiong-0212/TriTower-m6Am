import os
import sys
import gzip
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"

GENCODE_FASTA = DATA_DIR / "gencode_transcripts.fa"
GENCODE_FASTA_GZ = DATA_DIR / "gencode_transcripts.fa.gz"
GENCODE_V44_FASTA_GZ = DATA_DIR / "gencode.v44.transcripts.fa.gz"

TRAIN_FASTA = DATA_DIR / "train_ss_41.fasta"
TEST_FASTA = DATA_DIR / "test_ss_41.fasta"

OUTPUT_FASTA = DATA_DIR / "bca_negatives_41.fasta"
OUTPUT_STATS = DATA_DIR / "bca_negatives_stats.json"

BCA_B_DNA = set("CGT")
BCA_B_RNA = set("CGU")

WINDOW_HALF = 20
WINDOW_LEN = 41

TARGET_NEGATIVES = 32000


def read_ss_fasta_sequences(fasta_path):
    sequences = []
    with open(fasta_path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith('>'):
            seq_id = lines[i].strip()[1:]
            seq = lines[i + 1].strip().upper()
            sequences.append((seq_id, seq))
            i += 3
        else:
            i += 1
    return sequences


def load_known_positives():
    known = set()

    for fasta_path in [TRAIN_FASTA, TEST_FASTA]:
        if not fasta_path.exists():
            print(f"  [WARN] {fasta_path} not found, skipping")
            continue
        seqs = read_ss_fasta_sequences(fasta_path)
        for seq_id, seq in seqs:
            try:
                num = int(seq_id.replace('RNA', ''))
            except ValueError:
                continue
            is_positive = (num <= 3700) or (3700 < num <= 4020)
            if is_positive:
                known.add(seq)
        print(f"  Loaded {len(seqs)} sequences from {fasta_path.name}")

    print(f"  Total known positive sequences for exclusion: {len(known)}")
    return known


def dna_to_rna(seq):
    return seq.upper().replace('T', 'U')


def find_bca_sites(sequence):
    seq = sequence.upper()
    sites = []
    for i in range(len(seq) - 2):
        if seq[i] in BCA_B_DNA and seq[i + 1] == 'C' and seq[i + 2] == 'A':
            sites.append((i, i + 2))
    return sites


def extract_window(sequence, center_idx):
    start = center_idx - WINDOW_HALF
    end = center_idx + WINDOW_HALF + 1
    if start < 0 or end > len(sequence):
        return None
    return sequence[start:end]


def read_gencode_fasta(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    transcript_id = None
    seq_parts = []

    with opener(path, 'rt') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if transcript_id is not None and seq_parts:
                    yield transcript_id, ''.join(seq_parts)
                header = line[1:]
                transcript_id = header.split('|')[0].split('.')[0]
                seq_parts = []
            else:
                seq_parts.append(line.upper())
        if transcript_id is not None and seq_parts:
            yield transcript_id, ''.join(seq_parts)


def main():
    print("=" * 70)
    print("BCA Negative Sample Construction from GENCODE Transcriptome")
    print("=" * 70)

    gencode_path = None
    if GENCODE_FASTA.exists():
        gencode_path = GENCODE_FASTA
    elif GENCODE_FASTA_GZ.exists():
        gencode_path = GENCODE_FASTA_GZ
    elif GENCODE_V44_FASTA_GZ.exists():
        gencode_path = GENCODE_V44_FASTA_GZ
    else:
        print(f"\n[ERROR] GENCODE FASTA not found at:")
        print(f"  {GENCODE_FASTA}")
        print(f"  {GENCODE_FASTA_GZ}")
        print(f"  {GENCODE_V44_FASTA_GZ}")
        print(f"\nPlease download from:")
        print(f"  https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/")
        print(f"  gencode.v44.transcripts.fa.gz")
        print(f"\nAnd place it in: {DATA_DIR}/")
        sys.exit(1)

    print(f"\nGENCODE FASTA: {gencode_path}")

    print("\nLoading known m6Am positive samples for exclusion...")
    known_positives = load_known_positives()

    print(f"\nScanning transcriptome for BCA motifs...")
    print(f"  BCA motif: [C/G/U]CA, candidate site = A position")
    print(f"  Window: ±{WINDOW_HALF} nt ({WINDOW_LEN} nt total)")

    bca_negatives = []
    seen_windows = set()
    n_transcripts_scanned = 0
    n_bca_sites_found = 0
    n_excluded_known = 0
    n_excluded_duplicate = 0
    n_excluded_short = 0

    for transcript_id, dna_seq in read_gencode_fasta(gencode_path):
        n_transcripts_scanned += 1

        if n_transcripts_scanned % 10000 == 0:
            print(f"  Scanned {n_transcripts_scanned} transcripts, "
                  f"collected {len(bca_negatives)} BCA negatives so far...")

        bca_sites = find_bca_sites(dna_seq)
        n_bca_sites_found += len(bca_sites)

        for motif_start, candidate_idx in bca_sites:
            window_dna = extract_window(dna_seq, candidate_idx)
            if window_dna is None:
                n_excluded_short += 1
                continue

            window_rna = dna_to_rna(window_dna)

            if any(c not in 'ACGU' for c in window_rna):
                n_excluded_short += 1
                continue

            if window_rna in known_positives:
                n_excluded_known += 1
                continue

            if window_rna in seen_windows:
                n_excluded_duplicate += 1
                continue

            seen_windows.add(window_rna)
            bca_negatives.append({
                'id': f"BCA_{len(bca_negatives) + 1}",
                'transcript': transcript_id,
                'motif_pos': motif_start,
                'candidate_pos': candidate_idx,
                'sequence': window_rna,
            })

            if len(bca_negatives) >= TARGET_NEGATIVES * 2:
                pass

    print(f"\n{'='*70}")
    print(f"  BCA Scanning Summary")
    print(f"{'='*70}")
    print(f"  Transcripts scanned:     {n_transcripts_scanned:>10,}")
    print(f"  BCA sites found:         {n_bca_sites_found:>10,}")
    print(f"  Excluded (known m6Am):   {n_excluded_known:>10,}")
    print(f"  Excluded (duplicate):    {n_excluded_duplicate:>10,}")
    print(f"  Excluded (short/invalid):{n_excluded_short:>10,}")
    print(f"  Final BCA negatives:     {len(bca_negatives):>10,}")

    if len(bca_negatives) < TARGET_NEGATIVES:
        print(f"\n  [WARN] Only {len(bca_negatives)} BCA negatives found, "
              f"need {TARGET_NEGATIVES} for 1:100 ratio.")
        print(f"  Consider using a less strict BCA definition or scanning more transcripts.")

    print(f"\nWriting BCA negatives to: {OUTPUT_FASTA}")
    with open(OUTPUT_FASTA, 'w') as f:
        for i, neg in enumerate(bca_negatives):
            f.write(f">{neg['id']}\n")
            f.write(f"{neg['sequence']}\n")
            f.write(f"{'.' * WINDOW_LEN}\n")
    print(f"  Written {len(bca_negatives)} sequences (3-line format: header/seq/placeholder_ss)")

    import json
    stats = {
        'n_transcripts_scanned': n_transcripts_scanned,
        'n_bca_sites_found': n_bca_sites_found,
        'n_excluded_known_m6am': n_excluded_known,
        'n_excluded_duplicate': n_excluded_duplicate,
        'n_excluded_short_invalid': n_excluded_short,
        'n_final_bca_negatives': len(bca_negatives),
        'bca_motif_definition': 'B[C/G/U]A, candidate = A position',
        'window_length': WINDOW_LEN,
        'source': str(gencode_path.name),
    }
    with open(OUTPUT_STATS, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Statistics saved: {OUTPUT_STATS}")

    print(f"\n{'='*70}")
    print(f"  Next Steps")
    print(f"{'='*70}")
    print(f"  1. Run RNAfold on {OUTPUT_FASTA.name} to fill in secondary structure")
    print(f"     (replace placeholder '.' line with actual dot-bracket)")
    print(f"  2. Run RNA-FM embedding extraction on the FASTA")
    print(f"  3. Run eval_imbalanced.py for inference at 1:1, 1:10, 1:100 ratios")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
