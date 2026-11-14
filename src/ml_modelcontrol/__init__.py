from importlib import metadata

from ml_modelcontrol.export import export_python
from ml_modelcontrol.infer import infer
from ml_modelcontrol.model import GraphModel, SpecError, build_model
from ml_modelcontrol.registry import describe_units, get_unit
from ml_modelcontrol.spec import dump_spec, load_spec
from ml_modelcontrol.units import register_unit
from ml_modelcontrol.validate import validate

# バージョンは pyproject.toml の値を使用
try:
    __version__ = metadata.version("ml-modelcontrol")
except metadata.PackageNotFoundError:
    __version__ = "不明"

__all__ = [
    "GraphModel", "SpecError", "__version__", "build_model", "describe_units",
    "dump_spec", "export_python", "get_unit", "infer", "load_spec",
    "register_unit", "validate"
]
