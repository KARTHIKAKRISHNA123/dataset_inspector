"""Enable `python -m dataset_inspector ...` as an alias for the `dataset-inspect` console script."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
