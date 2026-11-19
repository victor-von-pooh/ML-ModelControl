from importlib import metadata

from ml_modelcontrol.custom import register_units_from
from ml_modelcontrol.export import export_python
from ml_modelcontrol.infer import infer
from ml_modelcontrol.model import GraphModel, SpecError, build_model
from ml_modelcontrol.pretrained import (
    classify_license, clear_pretrained_cache, pretrained_info
)
from ml_modelcontrol.registry import describe_units, get_unit
from ml_modelcontrol.spec import (
    dump_spec, embed_subgraph, list_examples, load_example, load_spec
)
from ml_modelcontrol.units import register_unit
from ml_modelcontrol.validate import describe_spec_units, validate

# バージョンは pyproject.toml の値を使用
try:
    __version__ = metadata.version("ml-modelcontrol")
except metadata.PackageNotFoundError:
    __version__ = "不明"

__all__ = [
    "GraphModel", "SpecError", "__version__", "build_model",
    "classify_license", "clear_pretrained_cache", "describe_spec_units",
    "describe_units", "dump_spec", "embed_subgraph", "export_python",
    "get_unit", "infer", "list_examples", "load_example", "load_spec",
    "pretrained_info", "register_unit", "register_units_from", "validate"
]
