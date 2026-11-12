"""Read-only Python startup stack forensic; no simulation state access."""
import faulthandler
import sys
faulthandler.enable(file=sys.stderr)
faulthandler.dump_traceback_later(95, repeat=False, file=sys.stderr)
