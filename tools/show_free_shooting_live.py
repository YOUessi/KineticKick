#!/usr/bin/env python3
"""Live entry point using the same controller, physics and evaluator as batch runs."""
import sys
from pathlib import Path

from eval_free_shooting import main

if __name__ == '__main__':
    if '--headless' in sys.argv:
        sys.argv.remove('--headless')
    else:
        sys.argv.extend(['--gui', '--hold'])
    if '--output' not in sys.argv:
        root = Path(__file__).resolve().parents[1]
        sys.argv.extend(['--output', str(root / 'reports/free_shooting_live.json')])
    main()
