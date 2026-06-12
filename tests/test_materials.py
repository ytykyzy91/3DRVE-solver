import json
from pathlib import Path

import numpy as np
import pytest

from rve_vam.materials import build_phase_mapping, isotropic_stiffness, load_json


def test_isotropic_stiffness_uses_engineering_shear():
    E = 10.0
    nu = 0.25
    C = isotropic_stiffness(E, nu)
    G = E / (2.0 * (1.0 + nu))
    assert C.shape == (6, 6)
    assert np.allclose(C, C.T)
    assert np.allclose(np.diag(C)[3:], G)


def test_example_material_mapping_is_one_based():
    config = load_json(Path(__file__).parents[1] / "example" / "user_RVE_analysis.json")
    mapping = build_phase_mapping(config, np.array([1, 2, 1, 2]))
    assert mapping.material_id_to_phase == {1: "matrix", 2: "reinforcement"}
    assert mapping.phase_to_material_name["reinforcement"] == "Glass"
    assert mapping.phase_to_material_name["matrix"] == "Epoxy"
    assert mapping.material_id_to_material[1].name == "Epoxy"
    assert mapping.material_id_to_material[2].name == "Glass"


def test_orthotropic_selected_material_is_not_implemented(tmp_path):
    config = load_json(Path(__file__).parents[1] / "example" / "user_RVE_analysis.json")
    config["Defs"]["Composite"]["Materials"]["reinforcement"] = "CFRP"
    with pytest.raises(NotImplementedError):
        build_phase_mapping(config, np.array([1, 2]))
