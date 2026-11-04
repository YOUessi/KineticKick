#!/usr/bin/env python3
"""Preview the corrected posture profile; target/power experiments keep v0.1.2."""
import sys
from pathlib import Path

from eval_free_shooting import main


def live_arguments(arguments):
    args = list(arguments)
    options = {arg.split('=', 1)[0] for arg in args}
    root = Path(__file__).resolve().parents[1]
    experimental = {'--target-x', '--power', '--speed-mps', '--targeted-acceptance',
                    '--baseline', '--trials', '--reference-sweep', '--grid', '--acceptance'}
    if '--config' not in options and not options.intersection(experimental):
        args.extend(['--config', str(root / 'configs/free_shooting_posture_v1.json')])
    if '--headless' in args:
        args.remove('--headless')
    else:
        args.extend(flag for flag in ('--gui', '--hold') if flag not in options)
    if '--output' not in options:
        args.extend(['--output', str(root / 'reports/free_shooting_live.json')])
    return args


if __name__ == '__main__':
    sys.argv[1:] = live_arguments(sys.argv[1:])
    main()
