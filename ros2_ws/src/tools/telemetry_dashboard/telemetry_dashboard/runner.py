#!/usr/bin/env python3

import sys
import signal
import atexit
from pathlib import Path

from telemetry_dashboard import launch_process_manager

def _stop_all():
    launch_process_manager.stop_all()


def main() -> int:
    atexit.register(_stop_all)

    def _handler(signum, _frame):
        _stop_all()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
    from streamlit.web import cli as stcli

    app_path = Path(__file__).with_name("app.py")
    sys.argv = ["streamlit", "run", str(app_path)]
    return stcli.main()


if __name__ == "__main__":
    raise SystemExit(main())
