import csv
import json

import numpy as np

from rve_vam.homogenization import HomogenizationResult
from rve_vam.materials import isotropic_stiffness
from rve_vam.postprocess import engineering_constants_from_stiffness, write_stiffness_csv, write_stiffness_json


def dummy_result(C):
    return HomogenizationResult(
        stiffness=C,
        stiffness_unsymmetrized=C,
        volume=1.0,
        phase_volume_fractions={"matrix": 1.0},
        solver_residuals=[0.0] * 6,
        material_mapping={"1": {"phase": "matrix", "material": "mat"}},
        diagnostics={},
    )


def test_engineering_constants_for_isotropic_stiffness():
    C = isotropic_stiffness(10.0, 0.25)
    constants = engineering_constants_from_stiffness(C)
    assert np.isclose(constants["E1"], 10.0)
    assert np.isclose(constants["E2"], 10.0)
    assert np.isclose(constants["E3"], 10.0)
    assert np.isclose(constants["nu12"], 0.25)
    assert np.isclose(constants["nu13"], 0.25)
    assert np.isclose(constants["nu23"], 0.25)
    assert np.isclose(constants["G12"], 4.0)
    assert np.isclose(constants["G23"], 4.0)
    assert np.isclose(constants["G13"], 4.0)


def test_stiffness_outputs_include_engineering_constants(tmp_path):
    result = dummy_result(isotropic_stiffness(10.0, 0.25))
    json_path = tmp_path / "stiffness.json"
    csv_path = tmp_path / "stiffness.csv"
    write_stiffness_json(result, json_path)
    write_stiffness_csv(result, csv_path)

    payload = json.loads(json_path.read_text())
    assert "engineering_constants" in payload
    assert np.isclose(payload["engineering_constants"]["E1"], 10.0)

    text = csv_path.read_text()
    assert "Engineering constants" in text
    assert "E1" in text
    assert "nu12" in text
