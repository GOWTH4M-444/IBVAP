"""
Add a person of interest to the face watchlist.

  python scripts/enroll_face.py POI_001 --name "Test Person" --category "Demo" --risk HIGH photo1.jpg photo2.jpg

Use only photos you are authorised to hold. Creates data/watchlist/<ID>/ and updates meta.json.
"""
import argparse
import json
import os
import shutil

p = argparse.ArgumentParser()
p.add_argument("poi_id")
p.add_argument("photos", nargs="+")
p.add_argument("--name", required=True)
p.add_argument("--category", default="")
p.add_argument("--risk", default="HIGH", choices=["LOW", "MEDIUM", "HIGH", "CRITICAL"])
p.add_argument("--unit", default="")
p.add_argument("--dir", default="data/watchlist")
a = p.parse_args()

os.makedirs(os.path.join(a.dir, a.poi_id), exist_ok=True)
for ph in a.photos:
    shutil.copy(ph, os.path.join(a.dir, a.poi_id, os.path.basename(ph)))
meta_path = os.path.join(a.dir, "meta.json")
meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
meta[a.poi_id] = {"name": a.name, "category": a.category, "risk_level": a.risk, "issuing_unit": a.unit}
json.dump(meta, open(meta_path, "w"), indent=2)
print(f"Enrolled {a.poi_id} with {len(a.photos)} photo(s). Total POIs: {len(meta)}")
