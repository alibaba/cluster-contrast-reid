from pathlib import Path
import json, shutil, random
from PIL import Image

# ---------- paths ----------
raw_root   = Path("/Users/armina/Downloads/NDD20/BELOW")          # folder with 1.jpg, 2.jpg, ...
anno_file  = raw_root.parent / "BELOW_LABELS.json"
out_root   = Path("examples/data/ndd/dolphin")
train_dir  = out_root / "bounding_box_train"
query_dir  = out_root / "query"
gallery_dir= out_root / "bounding_box_test"

for d in [train_dir, query_dir, gallery_dir]:
    d.mkdir(parents=True, exist_ok=True)

# ---------- parse annotations ----------
anno = json.loads(anno_file.read_text())

records = []   # (pid, frame_path, poly_index, bbox) for later split
for entry in anno.values():
    frame_name = entry["filename"]
    frame_path = raw_root / frame_name
    regions    = entry["regions"]

    for idx, reg in enumerate(regions):
        attrs = reg["region_attributes"]
        pid   = attrs.get("id")
        if pid is None or not pid.isdigit():   # skip unlabeled/background
            continue

        # polygon → tight bbox
        xs = reg["shape_attributes"]["all_points_x"]
        ys = reg["shape_attributes"]["all_points_y"]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        bbox = (xmin, ymin, xmax, ymax)

        records.append((int(pid), frame_path, idx, bbox))

print(f"Found {len(records)} dolphin instances in {len(anno)} frames.")

# ---------- crop & save ----------
camid = 1   # single-camera assumption; change if you have >1 viewpoints
for pid, frame_path, idx, bbox in records:
    with Image.open(frame_path) as im:
        crop = im.crop(bbox)
        pid_str   = f"{pid:04d}"
        frame_id  = frame_path.stem.zfill(6)        # e.g. '000567'
        det_id    = f"{idx:02d}"
        new_name  = f"{pid_str}_c{camid}s1_{frame_id}_{det_id}.jpg"
        crop.save(train_dir / new_name, quality=95)

print("Crops saved to bounding_box_train/")

# ---------- simple query / gallery split ----------
random.seed(0)
pid_to_files = {}
for f in train_dir.glob("*.jpg"):
    pid = int(f.name[:4])
    pid_to_files.setdefault(pid, []).append(f)

for pid, files in pid_to_files.items():
    files.sort()
    query   = files[0:1]                     # first crop → query
    gallery = files[1:] if len(files) > 1 else files  # rest → gallery

    for f in query:
        shutil.copy2(f, query_dir / f.name)
    for f in gallery:
        shutil.copy2(f, gallery_dir / f.name)

print("Created query/ and bounding_box_test/ splits.")