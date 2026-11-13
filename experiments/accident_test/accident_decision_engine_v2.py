
import csv
import json
from pathlib import Path

BASE = Path("experiments/accident_test")

VIDEOS = {
    "headon-nn": 1.54,
    "tbone": 0.20
}


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(value):
    try:
        return float(value) if value not in ("", None) else None
    except (ValueError, TypeError):
        return None


def analyse(name, candidate):
    prefix = BASE / f"{name}_v6"

    tracks = read_csv(Path(str(prefix) + "_tracks.csv"))
    pairs = read_csv(Path(str(prefix) + "_pairs.csv"))
    timeline = read_csv(Path(str(prefix) + "_timeline.csv"))

    score = 0
    evidence = []
    warnings = []

    # 1. Data coverage
    movement_tracks = 0
    speed_drops = 0
    direction_changes = 0

    for t in tracks:
        before_n = num(t["before_event_observations"]) or 0
        after_n = num(t["after_event_observations"]) or 0

        if before_n < 3 or after_n < 3:
            continue

        movement_tracks += 1
        before = num(t["before_pixel_speed"])
        after = num(t["after_pixel_speed"])
        direction = num(t["direction_change_degrees"])

        if before is not None and after is not None and before > 0:
            if (before - after) / before >= 0.30:
                speed_drops += 1

        if direction is not None and direction >= 30:
            direction_changes += 1

    if speed_drops:
        score += 20
        evidence.append(f"Pixel-speed reduction in {speed_drops} track(s)")

    if direction_changes:
        score += 20
        evidence.append(
            f"Direction change >=30 degrees in {direction_changes} track(s)"
        )

    # 2. Remove possible duplicate tracking pairs
    valid_pairs = []
    duplicate_pairs = []

    for p in pairs:
        iou = num(p["maximum_iou"]) or 0
        distance = num(p["minimum_distance_px"])

        if iou >= 0.70 and distance is not None and distance <= 25:
            duplicate_pairs.append(
                f'{p["track_a"]}-{p["track_b"]}'
            )
        else:
            valid_pairs.append(p)

    if duplicate_pairs:
        warnings.append(
            "Possible duplicate IDs excluded: " + ", ".join(duplicate_pairs)
        )

    # 3. Pair-distance reduction and overlap
    convergence = 0
    overlaps = 0

    for p in valid_pairs:
        before_n = num(p["before_event_observations"]) or 0
        after_n = num(p["after_event_observations"]) or 0

        before = num(p["before_mean_distance_px"])
        after = num(p["after_mean_distance_px"])

        if (
            before_n >= 3 and after_n >= 3
            and before is not None and after is not None
            and before > 0
        ):
            if (before - after) / before >= 0.20:
                convergence += 1

        iou = num(p["maximum_iou"]) or 0
        overlap_n = num(p["overlap_observations"]) or 0

        if iou >= 0.10 and overlap_n >= 3:
            overlaps += 1

    if convergence:
        score += 20
        evidence.append(
            f"Pair distance reduction in {convergence} pair(s)"
        )

    if overlaps:
        score += 20
        evidence.append(
            f"Bounding-box overlap in {overlaps} pair(s)"
        )

    # 4. Candidate-time timeline
    event_row = next(
        (
            row for row in timeline
            if num(row["relative_time_sec"]) == 0
        ),
        None
    )

    if event_row:
        event_overlap = num(event_row["overlap_observations"]) or 0
        if event_overlap >= 2:
            score += 10
            evidence.append(
                "Multiple box overlaps at candidate-time sample"
            )
    else:
        warnings.append("Candidate-time timeline sample missing")

    # 5. Reliability and temporal coverage
    reliable_tracks = sum(
        1 for t in tracks if (num(t["observations"]) or 0) >= 5
    )

    sufficient_coverage = candidate >= 1.0 and movement_tracks >= 2

    if candidate < 1.0:
        warnings.append(
            "Less than 1 second of pre-event footage; "
            "before/after evidence is unreliable"
        )

    if movement_tracks < 2:
        warnings.append(
            "Fewer than 2 tracks have enough before-and-after observations"
        )

    if reliable_tracks < 2:
        warnings.append("Insufficient reliable vehicle tracks")

    if not sufficient_coverage:
        status = "REVIEW_REQUIRED"
        evidence_level = "LIMITED"
    elif score >= 50:
        status = "POSSIBLE"
        evidence_level = "MULTIPLE_INDICATORS"
    else:
        status = "INSUFFICIENT_EVIDENCE"
        evidence_level = "LIMITED"

    result = {
        "video": name,
        "accident_status": status,
        "accident_type": "UNKNOWN",
        "candidate_time_seconds": candidate,
        "evidence_score": score,
        "evidence_level": evidence_level,
        "reliable_tracks": reliable_tracks,
        "tracks_with_before_after_data": movement_tracks,
        "total_vehicle_pairs": len(pairs),
        "valid_vehicle_pairs": len(valid_pairs),
        "excluded_duplicate_pairs": duplicate_pairs,
        "evidence": evidence,
        "warnings": warnings,
        "note": (
            "Exploratory rule-based indicators only. "
            "Score is not a probability or confirmed accident."
        )
    }

    output = BASE / f"{name}_decision_v2.json"
    output.write_text(json.dumps(result, indent=4), encoding="utf-8")

    print("\n" + "=" * 55)
    print("VIDEO:", name)
    print(json.dumps(result, indent=4))
    print("Saved:", output)


if __name__ == "__main__":
    for video, event_time in VIDEOS.items():
        analyse(video, event_time)

    print("\nDECISION ENGINE V2 COMPLETED.")
