#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import sys

def main():
    from AlphaPose_OSNet_Pipeline.Pipeline.pipeline import main as pipeline_main
    pipeline_main()

if __name__ == "__main__":
    main()
