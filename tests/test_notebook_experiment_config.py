"""Regression contract: one editable experiment cell per model notebook."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    "nllb": "01_nllb_research.ipynb",
    "indictrans2": "03_indictrans2_adaptation.ipynb",
    "bhashaverse": "04_bhashaverse_adaptation.ipynb",
}


def _configuration(nb):
    indices = []
    configs = []
    for index, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        parsed = ast.parse("".join(cell["source"]))
        for node in parsed.body:
            if (isinstance(node, ast.Assign) and
                    any(isinstance(target, ast.Name) and target.id == "EXPERIMENT"
                        for target in node.targets)):
                indices.append(index)
                configs.append(ast.literal_eval(node.value))
    assert len(configs) == 1, "Only ONE editable EXPERIMENT assignment allowed"
    assert indices[0] <= 3, "Settings must be near the top of the notebook"
    return configs[0]


def test_experiment_flags_are_single_source_of_truth():
    for family, filename in NOTEBOOKS.items():
        nb = json.loads((ROOT / "notebooks" / filename).read_text(encoding="utf-8"))
        cfg = _configuration(nb)
        assert cfg["data_mode"] == "aligned_text"
        assert cfg["validate_data"] is False
        assert cfg["data_report"] is False
        assert cfg["stage"] == "lowest_baseline"
        assert cfg["result_plots"] is True
        assert cfg["search_metric"] == "chrfpp"
        assert len(cfg["search_grid"]) >= 2
        assert cfg["split_ratios"] == (0.8, 0.1, 0.1) or cfg["split_ratios"] == [0.8, 0.1, 0.1]
        assert cfg["max_steps"] == -1
        assert cfg["include_checkpoints"] is False
        assert cfg["save_archive"] is True
        assert cfg["conditioning"] == ("text_prefix" if family == "bhashaverse" else "native")


def test_first_real_nllb_run_defaults():
    nb = json.loads((ROOT / "notebooks" / NOTEBOOKS["nllb"]).read_text())
    cfg = _configuration(nb)
    assert cfg["source_lang"] == "bho_Deva"
    assert cfg["target_lang"] == "hin_Deva"
    assert cfg["finetune_mode"] == "full"
    assert cfg["allow_training"] is False
    assert cfg["epochs"] == 3
    assert cfg["setup_mode"] == "train"
    assert cfg["extend_language"] is None
    assert cfg["proxy_source_lang"] is None
    assert cfg["proxy_target_lang"] is None
