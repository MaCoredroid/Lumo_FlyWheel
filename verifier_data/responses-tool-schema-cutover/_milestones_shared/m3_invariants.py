#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys

SLOT = 'M3_invariants'
result_file = os.environ.get("RESULT_FILE")
if not result_file:
    sys.exit(2)
payload = json.loads(open(result_file).read())
sys.exit(0 if payload.get("milestones", {}).get(SLOT) else 1)
