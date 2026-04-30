"""
run_app.py
----------
Entry-point launcher for the PSX40 Stock Screener.
Run this instead of typing the full streamlit command manually.

Usage
-----
    python run_app.py               # default port 8501
    python run_app.py --port 8080   # custom port
    python run_app.py --debug       # enable Streamlit debug mode
"""

import subprocess
import sys
import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the PSX40 Stock Screener Streamlit app."
    )
    parser.add_argument(
        "--port", type=int, default=8501,
        help="Port to run Streamlit on (default: 8501)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable Streamlit debug / development mode",
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="Do not automatically open the browser",
    )
    return parser.parse_args()


def check_env() -> None:
    """Warn if .env is missing; create from example if possible."""
    env_file     = Path(".env")
    env_example  = Path(".env.example")

    if not env_file.exists():
        if env_example.exists():
            import shutil
            shutil.copy(env_example, env_file)
            print("[run_app] .env not found — created from .env.example ✅")
        else:
            print("[run_app] ⚠️  No .env file found. Using default settings.")


def check_data_dir() -> None:
    """Ensure the data/ directory exists before Streamlit boots."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)


def main() -> None:
    args = parse_args()

    check_env()
    check_data_dir()

    app_path = Path("app") / "streamlit_app.py"
    if not app_path.exists():
        print(f"[run_app] ❌ App file not found at {app_path}")
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", str(args.port),
        "--server.headless", "true" if args.no_browser else "false",
        "--theme.base", "dark",
        "--theme.primaryColor", "#00e5a0",
        "--theme.backgroundColor", "#0a0e1a",
        "--theme.secondaryBackgroundColor", "#111827",
        "--theme.textColor", "#c9d1e0",
    ]

    if args.debug:
        cmd += ["--logger.level", "debug"]

    print(f"""
╔══════════════════════════════════════╗
║   PSX40 Stock Screener — Phase 1    ║
╠══════════════════════════════════════╣
║  URL  : http://localhost:{args.port:<11} ║
║  App  : {str(app_path):<28} ║
║  Debug: {'ON ' if args.debug else 'OFF':<29} ║
╚══════════════════════════════════════╝
""")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[run_app] Server stopped.")
    except subprocess.CalledProcessError as e:
        print(f"[run_app] ❌ Streamlit exited with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()