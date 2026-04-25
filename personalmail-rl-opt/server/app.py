"""
OpenEnv-facing server module.

This wrapper exposes `app` from the existing root `server.py` so OpenEnv validators
can discover the expected `server/app.py` path without changing runtime behavior.
"""

from pathlib import Path
import importlib.util


_ROOT_SERVER_PATH = Path(__file__).resolve().parent.parent / "server.py"
_spec = importlib.util.spec_from_file_location("personalmail_root_server", _ROOT_SERVER_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Unable to load root server module at {_ROOT_SERVER_PATH}")

_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
app = _module.app


def main():
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=7863)


if __name__ == "__main__":
    main()

