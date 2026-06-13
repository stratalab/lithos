"""Shared utilities: config loading, seeding, device/dtype, atomic I/O, checks.

Submodules import heavy optional deps (e.g. torch) lazily, so importing this
package stays cheap for fast unit tests.
"""
