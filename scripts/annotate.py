"""Start the private annotation page on this computer."""

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn

from annotation.app import create_app
from annotation.workspace import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    uvicorn.run(create_app(args.data_dir), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
