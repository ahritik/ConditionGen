#!/usr/bin/env python3
# utils/build_cond_map.py
import json, argparse, os

ap = argparse.ArgumentParser()
ap.add_argument("--label_map", required=True)
ap.add_argument("--train_log", required=True)  # unused for your format but kept for CLI compatibility
ap.add_argument("--out", default="cond_map.json")
args = ap.parse_args()

with open(args.label_map) as f:
    lm = json.load(f)

# Expect: {"artifact_names": ["none","eye","muscle","chewing","shiver","electrode"]}
arts = lm.get("artifact_names")
if not isinstance(arts, list) or not all(isinstance(s, str) for s in arts):
    raise SystemExit("Expected label_map.json to have key 'artifact_names' as a list of strings.")

cond = {
  "arts": arts,
  "schema": "[artifact_onehot | intensity]",
  "has_intensity": True,
  "has_seizure": False,
  "age_bins": 1,
  "montages": 1,
  "artifact_offset": 0
}

with open(args.out, "w") as f:
    json.dump(cond, f, indent=2)

print("Wrote", os.path.abspath(args.out))
print("Artifacts order:", arts)
