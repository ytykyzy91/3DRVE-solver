from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .utils import parse_bool

VOIGT_ORDER = ["xx", "yy", "zz", "xy", "yz", "xz"]
PHASE_ORDER = ["reinforcement", "matrix", "interphase"]


@dataclass(frozen=True)
class IsotropicMaterial:
    name: str
    E: float
    nu: float
    stiffness: np.ndarray


@dataclass(frozen=True)
class PhaseMapping:
    material_id_to_phase: dict[int, str]
    phase_to_material_name: dict[str, str]
    material_id_to_material: dict[int, IsotropicMaterial]

    def as_metadata(self) -> dict[str, dict[str, str]]:
        return {
            str(mid): {
                "phase": phase,
                "material": self.phase_to_material_name[phase],
            }
            for mid, phase in sorted(self.material_id_to_phase.items())
        }


def isotropic_stiffness(E: float, nu: float) -> np.ndarray:
    if E <= 0.0:
        raise ValueError(f"Young's modulus must be positive, got {E}.")
    if not (-1.0 < nu < 0.5):
        raise ValueError(f"Poisson ratio must be in (-1, 0.5), got {nu}.")
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    c = np.zeros((6, 6), dtype=float)
    c[:3, :3] = lam
    np.fill_diagonal(c[:3, :3], lam + 2.0 * mu)
    c[3, 3] = mu
    c[4, 4] = mu
    c[5, 5] = mu
    return c


def load_json(path: Path | str) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def analysis_materials_by_name(config: dict) -> dict[str, dict]:
    materials = config.get("Defs", {}).get("Analysis", {}).get("Materials")
    if not isinstance(materials, list):
        raise ValueError("Expected Defs.Analysis.Materials to be a list.")
    result: dict[str, dict] = {}
    for material in materials:
        name = material.get("name")
        if name:
            result[str(name)] = material
    return result


def composite_phase_material_names(config: dict) -> dict[str, str]:
    composite = config.get("Defs", {}).get("Composite", {}).get("Materials")
    if not isinstance(composite, dict):
        raise ValueError("Expected Defs.Composite.Materials to be an object.")

    names = {
        "reinforcement": str(composite.get("reinforcement")),
        "matrix": str(composite.get("matrix")),
    }
    interphase = composite.get("Interphase", {})
    if isinstance(interphase, dict) and parse_bool(interphase.get("enabled")):
        names["interphase"] = str(interphase.get("material"))
    return names


def material_from_definition(definition: dict) -> IsotropicMaterial:
    name = str(definition.get("name"))
    linear = definition.get("Mechanical", {}).get("LinearElastic", {})
    isotropic = linear.get("Isotropic")
    if isinstance(isotropic, dict) and parse_bool(isotropic.get("enabled", "1")):
        E = float(isotropic["E"])
        nu = float(isotropic["nu"])
        return IsotropicMaterial(name=name, E=E, nu=nu, stiffness=isotropic_stiffness(E, nu))

    orthotropic = linear.get("Orthotropic")
    if isinstance(orthotropic, dict) and parse_bool(orthotropic.get("enabled", "1")):
        raise NotImplementedError(
            f"Material {name!r} is orthotropic. The current solver implements isotropic linear elasticity only."
        )
    raise ValueError(f"Material {name!r} has no enabled isotropic linear elastic definition.")


def active_phase_order(phase_to_material_name: dict[str, str]) -> list[str]:
    return [phase for phase in PHASE_ORDER if phase in phase_to_material_name]


def auto_material_id_to_phase(material_ids: np.ndarray, phases: list[str]) -> dict[int, str]:
    ids = sorted(int(v) for v in np.unique(material_ids))
    phase_set = set(phases)

    if ids and min(ids) >= 1:
        one_based_priority = {
            1: "matrix",
            2: "reinforcement",
            3: "interphase",
        }
        mapping = {mid: one_based_priority[mid] for mid in ids if mid in one_based_priority}
        if len(mapping) == len(ids) and set(mapping.values()).issubset(phase_set):
            return mapping
        raise ValueError(
            f"Cannot auto-map one-based material IDs {ids} to active phases {phases}. "
            "Expected 1:matrix, 2:reinforcement, 3:interphase. Provide an explicit material ID map."
        )

    n = len(phases)
    if ids == list(range(n)):
        return dict(zip(ids, phases))
    raise ValueError(
        f"Ambiguous material IDs {ids}; expected zero-based {list(range(n))} or one-based IDs "
        "1:matrix, 2:reinforcement, 3:interphase. Provide an explicit material ID map."
    )


def parse_material_id_map(text: str | None) -> dict[int, str] | None:
    if not text:
        return None
    result: dict[int, str] = {}
    for item in text.split(","):
        key, sep, value = item.partition(":")
        if not sep:
            raise ValueError(f"Invalid material ID mapping item {item!r}; expected ID:phase.")
        phase = value.strip().lower()
        if phase not in PHASE_ORDER:
            raise ValueError(f"Invalid phase {phase!r}; expected one of {PHASE_ORDER}.")
        result[int(key.strip())] = phase
    return result


def build_phase_mapping(
    config: dict,
    material_ids: np.ndarray,
    explicit_id_map: dict[int, str] | None = None,
) -> PhaseMapping:
    phase_names = composite_phase_material_names(config)
    phases = active_phase_order(phase_names)
    material_id_to_phase = explicit_id_map or auto_material_id_to_phase(material_ids, phases)

    unknown_phases = sorted(set(material_id_to_phase.values()) - set(phase_names))
    if unknown_phases:
        raise ValueError(f"Material ID map references inactive or undefined phases: {unknown_phases}.")

    material_defs = analysis_materials_by_name(config)
    phase_to_material: dict[str, IsotropicMaterial] = {}
    for phase, material_name in phase_names.items():
        if material_name not in material_defs:
            raise ValueError(f"Phase {phase!r} references unknown material {material_name!r}.")
        phase_to_material[phase] = material_from_definition(material_defs[material_name])

    material_id_to_material = {
        int(mid): phase_to_material[phase] for mid, phase in material_id_to_phase.items()
    }
    return PhaseMapping(
        material_id_to_phase={int(k): v for k, v in material_id_to_phase.items()},
        phase_to_material_name=phase_names,
        material_id_to_material=material_id_to_material,
    )
