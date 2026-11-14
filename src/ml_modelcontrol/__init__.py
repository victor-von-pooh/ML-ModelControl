from importlib import metadata

# バージョンは pyproject.toml の値を使用
try:
    __version__ = metadata.version("ml-modelcontrol")
except metadata.PackageNotFoundError:
    __version__ = "不明"

__all__ = ["__version__"]
