# UNSW-NB15

**Source:** Australian Centre for Cyber Security, UNSW Canberra
**Official page:** https://research.unsw.edu.au/projects/unsw-nb15-dataset

The feature description file used for this work is `NUSW-NB15_features.csv`
(official feature-name/type/description reference sheet).

## Expected local layout
\`\`\`
raw/UNSW-NB15/*.csv
raw/UNSW-NB15/NUSW-NB15_features.csv
\`\`\`

## Preprocessing
\`\`\`bash
python preprocessing/preprocessing.py --dataset UNSW-NB15 --input raw/UNSW-NB15/ --output processed/UNSW-NB15/
\`\`\`

## License / Usage terms
Distributed by UNSW Canberra Cyber under its own terms; consult the official page.