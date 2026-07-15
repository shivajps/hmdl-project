# TON_IoT

**Source:** Australian Centre for Cyber Security, UNSW Canberra
**Official page:** https://research.unsw.edu.au/projects/toniot-datasets

Network-flow subset used in this work: `train_test_network.csv`.

## Expected local layout
\`\`\`
raw/TON_IoT/train_test_network.csv
\`\`\`

## Preprocessing
\`\`\`bash
python preprocessing/preprocessing.py --dataset TON_IoT --input raw/TON_IoT/ --output processed/TON_IoT/
\`\`\`

## License / Usage terms
Distributed by UNSW Canberra Cyber under its own terms; consult the official page.