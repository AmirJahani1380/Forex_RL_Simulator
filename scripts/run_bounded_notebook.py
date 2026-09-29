"""Execute the bounded notebook cells and save their actual text outputs."""

import base64
import contextlib
import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Headless execution saves open figures below; notebook users still see plt.show().
plt.show = lambda: None

path = Path(__file__).resolve().parents[1] / "Forex_RL_Agent_Final.ipynb"
sys.path.insert(0, str(path.parent / "src"))
notebook = json.loads(path.read_text(encoding="utf-8"))
scope = {"__name__": "__main__"}

for count, cell in enumerate((cell for cell in notebook["cells"] if cell["cell_type"] == "code"), 1):
    output = io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        exec(compile("".join(cell["source"]), f"cell-{count}", "exec"), scope)
    cell["outputs"] = (
        [{"output_type": "stream", "name": "stdout", "text": output.getvalue().splitlines(keepends=True)}]
        if output.getvalue()
        else []
    )
    for number in plt.get_fignums():
        figure = plt.figure(number)
        image = io.BytesIO()
        figure.savefig(image, format="png", dpi=110, bbox_inches="tight")
        cell["outputs"].append(
            {
                "output_type": "display_data",
                "data": {"image/png": base64.b64encode(image.getvalue()).decode("ascii")},
                "metadata": {},
            }
        )
        plt.close(figure)
    cell["execution_count"] = count
    print(f"cell {count} passed")

path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
