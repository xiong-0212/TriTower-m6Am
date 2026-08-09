import os
import sys
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.resolve()


def run_rnafold_batch(sequences, batch_size=500):
    all_ss = []
    for start in tqdm(range(0, len(sequences), batch_size),
                      desc="Running RNAfold"):
        batch = sequences[start:start + batch_size]

        input_text = '\n'.join(batch) + '\n'

        result = subprocess.run(
            ['RNAfold', '--noPS'],
            input=input_text,
            capture_output=True,
            text=True,
            check=True,
        )

        lines = result.stdout.strip().split('\n')
        for i in range(0, len(lines), 2):
            if i + 1 < len(lines):
                ss_line = lines[i + 1].strip()
                ss = ss_line.split()[0]
                all_ss.append(ss)

    return all_ss


def main():
    parser = argparse.ArgumentParser(description="Run RNAfold on BCA negatives")
    parser.add_argument('--input', type=str,
                        default=str(PROJECT_ROOT / "data" / "bca_negatives_41_sampled.fasta"),
                        help='Input 3-line FASTA (header/seq/placeholder)')
    parser.add_argument('--output', type=str,
                        default=str(PROJECT_ROOT / "data" / "bca_negatives_41_sampled_ss.fasta"),
                        help='Output 3-line FASTA (header/seq/dot-bracket)')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='Batch size for RNAfold')
    args = parser.parse_args()

    print("=" * 60)
    print("RNAfold for BCA Negative Samples")
    print("=" * 60)

    try:
        result = subprocess.run(['RNAfold', '--version'],
                                 capture_output=True, text=True)
        print(f"  RNAfold version: {result.stdout.strip()}")
    except FileNotFoundError:
        print(f"\n[ERROR] RNAfold not found. Please install ViennaRNA:")
        print(f"  conda install -c bioconda viennarna")
        sys.exit(1)

    print(f"\nReading: {args.input}")
    headers = []
    seqs = []
    with open(args.input) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith('>'):
            headers.append(lines[i].strip())
            seqs.append(lines[i + 1].strip().upper())
            i += 3
        else:
            i += 1
    print(f"  Loaded {len(seqs)} sequences")

    print(f"\nRunning RNAfold (batch_size={args.batch_size})...")
    ss_list = run_rnafold_batch(seqs, batch_size=args.batch_size)

    if len(ss_list) != len(seqs):
        print(f"[ERROR] Mismatch: {len(seqs)} sequences but {len(ss_list)} structures")
        sys.exit(1)

    for i, (seq, ss) in enumerate(zip(seqs, ss_list)):
        if len(ss) != len(seq):
            print(f"[WARN] Length mismatch at {i}: seq={len(seq)}, ss={len(ss)}")
            if len(ss) > len(seq):
                ss_list[i] = ss[:len(seq)]
            else:
                ss_list[i] = ss + '.' * (len(seq) - len(ss))

    print(f"\nWriting: {args.output}")
    with open(args.output, 'w') as f:
        for header, seq, ss in zip(headers, seqs, ss_list):
            f.write(f"{header}\n")
            f.write(f"{seq}\n")
            f.write(f"{ss}\n")
    print(f"  Written {len(seqs)} sequences with secondary structure")

    import shutil
    shutil.move(args.output, args.input)
    print(f"  Updated original file: {args.input}")
    print(f"\nDone! Next step: run RNA-FM extraction on the FASTA.")


if __name__ == '__main__':
    main()
