#!/bin/bash
# Judge-task verifier entrypoint. Writes harbor's reward file (1 = the judge
# produced a well-formed, grounded verdict; 0 = it did not). Judgment CORRECTNESS
# is evaluated later by the human-agreement study, not here.
set -uo pipefail
mkdir -p /logs/verifier

python3 -m pip install --quiet jsonschema >/dev/null 2>&1 || true
if python3 /tests/verify_verdict.py; then
  echo 1 > /logs/verifier/reward.txt
  exit 0
else
  echo 0 > /logs/verifier/reward.txt
  exit 1
fi
