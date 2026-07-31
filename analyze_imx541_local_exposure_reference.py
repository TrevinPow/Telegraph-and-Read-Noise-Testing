from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


BASE = Path(__file__).resolve().parent
ROOT = BASE / "analysis_work_emva1288_10V_maxcool" / "emva1288_10V_maxcool"
OUT = BASE / "imx541_emva1288_10V_maxcool_local_exposure_reference"
OUT.mkdir(exist_ok=True)
PATTERN = re.compile(r"_(bright|dark)_sn[^_]+_exp(\d+)us_n(\d+)\.tiff$")
TOLERANCES = (0.10, 0.25, 0.50)

groups: dict[str, list[tuple[int, list[tuple[int, Path]]]]] = {"dark": [], "bright": []}
for path in ROOT.rglob("*.tiff"):
    match = PATTERN.search(path.name)
    if match:
        category, exposure, index = match.group(1), int(match.group(2)), int(match.group(3))
        key = (category, exposure)
        entry = next((item for item in groups[category] if item[0] == exposure), None)
        if entry is None:
            groups[category].append((exposure, [(index, path)]))
        else:
            entry[1].append((index, path))
for category in groups:
    groups[category].sort()
    for exposure, members in groups[category]:
        members.sort()
        if [index for index, _ in members] != list(range(8)):
            raise RuntimeError(f"Expected n00..n07 for {category} {exposure} us")


def read_residual(members: list[tuple[int, Path]]) -> np.ndarray:
    stack = np.stack([np.asarray(Image.open(path), dtype=np.float32) for _, path in members])
    return stack - stack.mean(axis=0, keepdims=True)


rows = []
for tolerance in TOLERANCES:
    for category in ("dark", "bright"):
        category_groups = groups[category]
        for exposure, target_members in category_groups:
            reference_groups = [
                (other_exposure, other_members)
                for other_exposure, other_members in category_groups
                if other_exposure != exposure
                and abs(other_exposure - exposure) / exposure <= tolerance
            ]
            target = read_residual(target_members).astype(np.float64)
            n_ref_frames = len(reference_groups) * 8
            if len(reference_groups) < 2:
                rows.append({
                    "relative_window": tolerance,
                    "category": category,
                    "exposure_us": exposure,
                    "reference_sets": len(reference_groups),
                    "reference_frames": n_ref_frames,
                    "positive_z_gt_4_38_pixel_frames": "",
                    "positive_z_gt_4_38_fraction": "",
                    "max_positive_z": "",
                })
                continue

            ref_sum = np.zeros(target.shape[1:], dtype=np.float64)
            ref_sum2 = np.zeros(target.shape[1:], dtype=np.float64)
            for _, members in reference_groups:
                ref = read_residual(members).astype(np.float64)
                ref_sum += ref.sum(axis=0)
                ref_sum2 += np.square(ref).sum(axis=0)
            ref_mean = ref_sum / n_ref_frames
            ref_var = (ref_sum2 - n_ref_frames * np.square(ref_mean)) / (n_ref_frames - 1)
            ref_std = np.sqrt(np.maximum(ref_var, 0.0))
            z = np.divide(target - ref_mean, ref_std, out=np.zeros_like(target), where=ref_std > 0)
            outlier = z > 4.38
            rows.append({
                "relative_window": tolerance,
                "category": category,
                "exposure_us": exposure,
                "reference_sets": len(reference_groups),
                "reference_frames": n_ref_frames,
                "positive_z_gt_4_38_pixel_frames": int(outlier.sum()),
                "positive_z_gt_4_38_fraction": float(outlier.mean()),
                "max_positive_z": float(z.max()),
            })

with (OUT / "per_set_local_exposure_summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

aggregate = {}
for tolerance in TOLERANCES:
    aggregate[str(tolerance)] = {}
    for category in ("dark", "bright"):
        valid = [
            row for row in rows
            if row["relative_window"] == tolerance
            and row["category"] == category
            and row["positive_z_gt_4_38_fraction"] != ""
        ]
        total_outliers = sum(row["positive_z_gt_4_38_pixel_frames"] for row in valid)
        total_tested = len(valid) * 8 * 1128 * 1128
        aggregate[str(tolerance)][category] = {
            "analyzed_sets": len(valid),
            "positive_z_gt_4_38_pixel_frames": total_outliers,
            "tested_pixel_frames": total_tested,
            "pooled_fraction": total_outliers / total_tested if total_tested else None,
            "pooled_percentage": 100 * total_outliers / total_tested if total_tested else None,
            "mean_per_set_fraction": float(np.mean([row["positive_z_gt_4_38_fraction"] for row in valid])) if valid else None,
            "median_reference_sets": float(np.median([row["reference_sets"] for row in valid])) if valid else None,
        }

summary = {
    "zip": "emva1288_10V_maxcool.zip",
    "z_threshold": 4.38,
    "window_definition": "Other sets whose exposure time is within the stated relative window of the target exposure, excluding the target set.",
    "residual_definition": "Each set is converted to frame-minus-that-set's pixel-wise 8-frame mean before local reference statistics are calculated.",
    "aggregate": aggregate,
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
