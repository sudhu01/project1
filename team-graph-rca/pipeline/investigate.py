"""Thin application entry point for frozen captured incidents."""
from rca_investigation.cli import main

if __name__=='__main__':
    raise SystemExit(main())
