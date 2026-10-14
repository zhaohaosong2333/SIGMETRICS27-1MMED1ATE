"""Model writer: serializes a model object into a .out file."""

from pathlib import Path
from ..models.copse import COPSEModel
from ..models.vesta import VESTAModel
from ..models.imm import IMMModel


def write_vector(vec: list, indent: int = 0) -> str:
    """Formats and writes a vector."""
    result = ", ".join(str(num) for num in vec)
    return " " * indent + "{" + result + "}"


def write_matrix(mat: list, indent: int = 0) -> str:
    """Formats and writes a matrxi."""
    rows = [write_vector(vec, indent) for vec in mat]
    return "{\n" + ",\n".join(rows) + "}"


def write_tensor(ten: list, indent: int = 0) -> str:
    """Formats and writes a tensor."""
    mats = [write_matrix(mat, indent) for mat in ten]
    return "{\n" + ",\n\n".join(mats) + "}"


def write_copse_model(model: COPSEModel, filepath: str):
    """Writes a COPSE model."""
    content = _format_copse_model(model)
    Path(filepath).write_text(content)


def write_vesta_model(model: VESTAModel, filepath: str):
    """Writes a VESTA model."""
    content = _format_vesta_model(model)
    Path(filepath).write_text(content)


def write_imm_model(model: IMMModel, filepath: str):
    """Writes an 1MM model."""
    content = _format_imm_model(model)
    Path(filepath).write_text(content)


def _format_copse_model(model: COPSEModel) -> str:
    """Formats a COPSE model as C++-style text."""
    lines = [
        f"auto {model.name}()",
        "{",
        f"    PtxtModelDescription ptxt_{model.name};",
        "",
        f'    ptxt_{model.name}.name = "{model.name}";',
        "",
        f"    ptxt_{model.name}.thresholds = {write_vector(model.thresh)};",
        "",
        f"    ptxt_{model.name}.bits = {model.p};",
        "",
        f"    ptxt_{model.name}.level_mask = {write_matrix(model.masks)};",
        "",
        f"    ptxt_{model.name}.level_b2s = {write_tensor(model.lvls)};",
        "",
        f"    ptxt_{model.name}.d2b = {write_matrix(model.reshuf)};",
        "",
        f"    ptxt_{model.name}.k = {model.k};",
        "",
        f"    return ptxt_{model.name};",
        "}"
    ]
    return "\n".join(lines)


def _format_vesta_model(model: VESTAModel) -> str:
    """Formats a VESTA model as C++-style text."""
    lines = [
        f"auto {model.name}()",
        "{",
        f"    PtxtModelDescription ptxt_{model.name};",
        "",
        f'    ptxt_{model.name}.name = "{model.name}";',
        "",
        f"    ptxt_{model.name}.thresholds = {write_vector(model.thresh)};",
        "",
        f"    ptxt_{model.name}.bits = {model.p};",
        "",
        f"    ptxt_{model.name}.level_mask = {write_matrix(model.masks)};",
        "",
        f"    ptxt_{model.name}.ulvls = {write_tensor(model.ulvls)};",
        "",
        f"    ptxt_{model.name}.k = {model.k};",
        "",
        f"    return ptxt_{model.name};",
        "}"
    ]
    return "\n".join(lines)


def _format_imm_model(model: IMMModel) -> str:
    """Formats a 1MM model as C++-style text."""
    lines = [
        f"auto {model.name}()",
        "{",
        f"    PtxtModelDescription ptxt_{model.name};",
        "",
        f'    ptxt_{model.name}.name = "{model.name}";',
        "",
        f"    ptxt_{model.name}.thresholds = {write_vector(model.dthresh)};",
        "",
        f"    ptxt_{model.name}.bits = {model.p};",
        "",
        f"    ptxt_{model.name}.em = {write_matrix(model.em)};",
        "",
        f"    ptxt_{model.name}.k = {model.k};",
        "",
        f"    return ptxt_{model.name};",
        "}"
    ]
    return "\n".join(lines)
