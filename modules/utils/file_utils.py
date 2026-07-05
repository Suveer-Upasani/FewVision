# modules/utils/file_utils.py
"""File and directory utilities for the FewVision pipeline."""

import os
import shutil
import uuid
import zipfile
from typing import Optional


def ensure_dir(path: str) -> str:
    """Create *path* and all parents if they do not exist.

    Parameters
    ----------
    path : str
        Directory to create.

    Returns
    -------
    str
        The same path (for chaining).
    """
    os.makedirs(path, exist_ok=True)
    return path


def ensure_data_dirs(base: str = "data") -> dict[str, str]:
    """Ensure all standard data sub-directories exist.

    Parameters
    ----------
    base : str
        Root data directory (default: ``"data"``).

    Returns
    -------
    dict[str, str]
        Mapping of ``{name: absolute_path}`` for each sub-directory.
    """
    dirs = {
        "uploads": os.path.join(base, "uploads"),
        "augmented": os.path.join(base, "augmented"),
        "reports": os.path.join(base, "reports"),
        "logs": os.path.join(base, "logs"),
        "temp": os.path.join(base, "temp"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def new_session_id() -> str:
    """Generate a unique session identifier.

    Returns
    -------
    str
        A short UUID4 hex string (12 characters).
    """
    return uuid.uuid4().hex[:12]


def clear_dir(path: str) -> None:
    """Remove all contents of *path* without deleting the directory itself.

    Parameters
    ----------
    path : str
        Directory to clear.
    """
    if not os.path.isdir(path):
        return
    for name in os.listdir(path):
        full = os.path.join(path, name)
        if os.path.isfile(full) or os.path.islink(full):
            os.remove(full)
        elif os.path.isdir(full):
            shutil.rmtree(full)


def create_zip(source_dir: str, output_path: Optional[str] = None) -> str:
    """Create a ZIP archive of *source_dir*.

    Parameters
    ----------
    source_dir : str
        Directory whose contents will be zipped.
    output_path : str, optional
        Destination ``.zip`` file path. Defaults to ``source_dir + ".zip"``.

    Returns
    -------
    str
        Absolute path to the created ZIP file.
    """
    if output_path is None:
        output_path = source_dir.rstrip(os.sep) + ".zip"

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(source_dir):
            for file in files:
                abs_path = os.path.join(root, file)
                arcname = os.path.relpath(abs_path, start=os.path.dirname(source_dir))
                zf.write(abs_path, arcname)

    return output_path


def safe_filename(name: str) -> str:
    """Return a filesystem-safe version of *name*.

    Strips path separators and other unsafe characters.

    Parameters
    ----------
    name : str
        Original filename.

    Returns
    -------
    str
        Safe filename string.
    """
    from werkzeug.utils import secure_filename as _sf
    return _sf(name)
