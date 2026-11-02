# ll-ml-triggerkit

Trainable low-level ML trigger chains for Cherenkov cameras (SST-1M / DigiCam).

Distribution name: `ll-ml-triggerkit`. Import name: `triggerkit`.

Python 3.12 or 3.13 (the hexagonal mapping of the camera comes from `dl1-data-handler`, which needs it).
The environment is described by `pyproject.toml` (Poetry); `poetry.lock` pins the versions the quantized models and their
hls4ml export were validated with (TensorFlow 2.20, Keras 3.12, HGQ2 0.1.9, hls4ml 1.3.0).

```bash
poetry install                    # core + tests (the exact locked versions)
poetry install -E qat             # + keras-hexagdly, HGQ2 and hls4ml (fully quantized models and their export)
poetry install -E hexcnn          # + hex CNN bodies only (keras-hexagdly)
```

`pip` works too (it ignores the lock, so the versions may differ): `pip install -e .`, `pip install -e '.[qat]'`.
On a GPU machine, add the CUDA libraries of TensorFlow: `pip install "tensorflow[and-cuda]==2.20.0"` inside the environment.

```python
import triggerkit
from triggerkit.TriggerChain import TriggerChain
```

## Layout

- `src/triggerkit/` — the library.
  - `TriggerChain.py` — core chain (`train_chain`, `compute_statistics`, `show_trigger_chain`, ...).
  - `Stages/ Loss/ Metric/ Callback/ Statistics/ FileIO/` — building blocks.
  - `data/ models/ training/` — dataset, pluggable bodies, training utilities (added incrementally).
  - Per-board trigger (B1X/B2X/B3X): `models/boards.py` (`CameraBoardSplitter`), the `ScatterToBoards` / `UnfoldBoards` / `BoardMax` layers and `TriggerDataset(split="board")`; see `examples/train_hexcnn_boards.py`.
- `examples/` — train / evaluate / show scripts.
- `tests/`

## Status

Bootstrapping: package extracted from the `python-reference` research sandbox and
renamed to `triggerkit`. Incremental refactor in progress (dataset, pluggable
model bodies, cascade statistics).

## Tests

```bash
poetry install
poetry run pytest   # ~200 tests, CPU only, several minutes (some spawn reader processes)
```
