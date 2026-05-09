import json, sys
sys.stdout.reconfigure(encoding='utf-8')

nb = json.load(open(r'e:\Thesis Claude\claude_msa_edl_co.ipynb', 'r', encoding='utf-8'))
cells = nb['cells']

# Print training log (cell 8) - first part
for o in cells[8].get('outputs', []):
    if o.get('output_type') == 'stream':
        text = ''.join(o.get('text', []))
        print(text[:3000])
        break

# Print cell 4 outputs (data stats)
print("\n\n=== CELL 4 (data stats) ===")
for o in cells[4].get('outputs', []):
    if o.get('output_type') == 'stream':
        print(''.join(o.get('text', [])))
