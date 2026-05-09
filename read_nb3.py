import json, sys
sys.stdout.reconfigure(encoding='utf-8')

nb = json.load(open(r'e:\Thesis Claude\claude_msa_edl_co.ipynb', 'r', encoding='utf-8'))

for o in nb['cells'][8].get('outputs', []):
    if o.get('output_type') == 'stream':
        text = ''.join(o.get('text', []))
        # Print epoch lines
        for line in text.split('\n'):
            if 'Epoch' in line and '|' in line:
                print(line.strip())
