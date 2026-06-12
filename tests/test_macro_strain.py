import json
from pathlib import Path

from rve_vam.config import FieldOutputOptions
from rve_vam.macro_strain import (
    analysis_with_internal_engineering_strain,
    macro_strain_from_legacy_material_config,
    parse_macro_strain_cli,
    parse_macro_strain_config,
)
from rve_vam.materials import load_json


def test_parse_macro_strain_cli():
    analysis = parse_macro_strain_cli(
        "0.1,0,0,0,0,0",
        load_steps=2,
        field_output=FieldOutputOptions(enabled=True),
    )
    assert analysis.load_steps == 2
    assert analysis.cases[0].strain == (0.1, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_parse_macro_strain_config(tmp_path):
    path = tmp_path / "macro.json"
    path.write_text(
        json.dumps(
            {
                "load_steps": 3,
                "output": {"write_vtu": True, "output_every": 2, "prefix": "m"},
                "cases": [{"name": "exx", "macro_strain": [0.01, 0, 0, 0, 0, 0]}],
            }
        )
    )
    analysis = parse_macro_strain_config(path)
    assert analysis.load_steps == 3
    assert analysis.field_output.output_every == 2
    assert analysis.cases[0].name == "exx"


def test_parse_legacy_macro_strain_mapping():
    config = {
        "Defs": {
            "Composite": {
                "LocalFieldsRecovery": {
                    "MacroStrain": {
                        "epsilon11": "0.01",
                        "epsilon22": "0",
                        "epsilon33": "0",
                        "epsilon12": "0.02",
                        "epsilon13": "0.03",
                        "epsilon23": "0.04",
                    }
                }
            }
        }
    }
    analysis = macro_strain_from_legacy_material_config(
        config,
        load_steps=1,
        field_output=FieldOutputOptions(enabled=True),
    )
    assert analysis.cases[0].strain == (0.01, 0.0, 0.0, 0.02, 0.04, 0.03)


def test_tensor_shear_input_is_converted_to_internal_engineering_shear():
    analysis = parse_macro_strain_cli(
        "0,0,0,0.01,0.02,0.03",
        load_steps=1,
        field_output=FieldOutputOptions(enabled=True),
        strain_convention="tensor_shear",
    )
    internal = analysis_with_internal_engineering_strain(analysis)
    assert analysis.cases[0].strain == (0.0, 0.0, 0.0, 0.01, 0.02, 0.03)
    assert internal.cases[0].strain == (0.0, 0.0, 0.0, 0.005, 0.01, 0.015)
