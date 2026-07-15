# CICIDS2017

**Source:** Canadian Institute for Cybersecurity, University of New Brunswick
**Official page:** https://www.unb.ca/cic/datasets/ids-2017.html

This work uses the Friday Working Hours (Afternoon) DDoS capture
(`Friday-WorkingHours-Afternoon-DDos_pcap_ISCX.csv`) as one of the
CICIDS2017 subsets referenced in Section 5.1/6.2 of the manuscript.

## Expected local layout
\`\`\`
raw/CICIDS2017/*.csv
\`\`\`

## Preprocessing
Run:
\`\`\`bash
python preprocessing/preprocessing.py --dataset CICIDS2017 --input raw/CICIDS2017/ --output processed/CICIDS2017/
\`\`\`

## License / Usage terms
CICIDS2017 is distributed by the Canadian Institute for Cybersecurity under its
own terms of use; consult the official page for citation and usage requirements.