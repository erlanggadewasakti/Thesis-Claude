import json, sys
sys.stdout.reconfigure(encoding='utf-8')

nb = json.load(open(r'e:\Thesis Claude\claude_msa_edl_co.ipynb', 'r', encoding='utf-8'))
cells = nb['cells']

for i, c in enumerate(cells):
    outs = c.get('outputs', [])
    ct = c.get('cell_type', '?')
    src_preview = ''.join(c.get('source', []))[:60].replace('\n', ' ')
    print(f"=== CELL {i} ({ct}) | {len(outs)} outputs | {src_preview}")
    
    for o in outs:
        otype = o.get('output_type', '?')
        if otype == 'stream':
            text = ''.join(o.get('text', []))
            print(text[-3000:] if len(text) > 3000 else text)
        elif otype in ('execute_result', 'display_data'):
            if 'text/plain' in o.get('data', {}):
                txt = ''.join(o['data']['text/plain'])
                print(txt[:2000])
        elif otype == 'error':
            print(f"ERROR: {''.join(o.get('traceback', []))[:500]}")
    print()
