# N-BaIoT

**Source:** UCI Machine Learning Repository
**Official page:** https://archive.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot

Benign traffic reference file used in this work: `1_benign.csv`.

## Expected local layout
\`\`\`
raw/N-BaIoT/1_benign.csv
raw/N-BaIoT/<device>_<attack>.csv
\`\`\`

## Preprocessing
\`\`\`bash
python preprocessing/preprocessing.py --dataset N-BaIoT --input raw/N-BaIoT/ --output processed/N-BaIoT/
\`\`\`

## License / Usage terms
Distributed via the UCI Machine Learning Repository under its own terms.