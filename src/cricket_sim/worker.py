"""Internal isolated-generation entry point; publication belongs to the parent process."""

import json
import sys


def main():
    job, settings = sys.argv[1], json.loads(sys.argv[2])
    if job == 'build':
        from .build import _build
        _build(**settings)
    elif job == 'model':
        from .models.run import _run
        _run(**settings)
    else:
        raise SystemExit(f'unknown internal job: {job}')


if __name__ == '__main__':
    main()
