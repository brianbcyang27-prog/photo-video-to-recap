"""Auto-edit a trip recap from an unorganised pile of photos and videos."""
from .analysis import Item, analyse_library  # noqa: F401
from .config import Analysis, Music, Pipeline, Render  # noqa: F401
from .context import attach_items, build_context  # noqa: F401
from .ingest import Library, scan  # noqa: F401
from .music import MusicResult, prepare_music  # noqa: F401
from .render import RenderResult, render  # noqa: F401
from .report import build_report  # noqa: F401
from .select import CutList, build_cutlist  # noqa: F401

__version__ = "1.0.0"
