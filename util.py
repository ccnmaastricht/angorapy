#!/usr/bin/env python
"""Helper functions."""
import sys


def flat_print(string: str):
    """A bit of a workaround to no new line printing to have it work in PyCharm."""
    print(f"\r{string}", end="")
