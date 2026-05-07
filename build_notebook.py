"""Build the complete notebook from cell source files."""
import json, os

NOTEBOOK_PATH = r"e:\Thesis Claude\claude_msa_edl_co.ipynb"
CELLS_DIR = r"e:\Thesis Claude\cells"

# Read existing notebook to preserve first 3 cells
with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

existing_cells = nb["cells"][:3]  # Keep markdown + config + data loading

# Read new cell files in order
new_cells = []
cell_files = sorted([f for f in os.listdir(CELLS_DIR) if f.endswith(".py")])

for fname in cell_files:
    path = os.path.join(CELLS_DIR, fname)
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.split("\n")
    }
    # Fix: each line except last needs \n
    cell["source"] = [line + "\n" for line in cell["source"][:-1]] + [cell["source"][-1]]
    new_cells.append(cell)

nb["cells"] = existing_cells + new_cells

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Built notebook with {len(nb['cells'])} cells ({len(existing_cells)} existing + {len(new_cells)} new)")
