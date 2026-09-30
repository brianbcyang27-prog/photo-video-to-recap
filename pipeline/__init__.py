"""Auto-edit a trip recap from an unorganised pile of photos and videos."""
from .config import Pipeline, Render, Analysis, Music  # noqa: F401
from .ingest import Library, scan  # noqa: F401
from .analysis import analyse_library, Item  # noqa: F401
from .context import build_context, attach_items  # noqa: F401
from .music import prepare_music, MusicResult  # noqa: F401
from .select import build_cutlist, CutList  # noqa: F401
from .render import render, RenderResult  # noqa: F401
from .report import build_report  # noqa: F401

__version__ = "1.0.0"
