# regressions/

Standalone, public-API scenario scripts. Each file exercises molnex the way a
user would (construct → configure → one concern → read result) and asserts
against **hard-coded golden literals** captured once, offline, from a named
commit — never from a live third-party oracle, network call, or subprocess.

These are **not** collected by pytest (`tests/` holds single-function unit
tests only; nothing here is imported by the suite). Run one directly:

```bash
PYTHONPATH=src python regressions/<name>.py   # prints OK, exits 0; exits 1 on drift
```

Every script's header comment must record the capture command, the commit sha
the goldens came from, the torch version, the date, and the device/precision.
