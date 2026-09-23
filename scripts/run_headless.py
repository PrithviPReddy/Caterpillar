"""Run a scenario end-to-end without the UI and print the event timeline.

usage: python scripts/run_headless.py [scenario] [--all] [--json out.jsonl]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detect.pipeline import DetectionPipeline  # noqa: E402
from sim.scenarios import get_scenario  # noqa: E402
from sim.world import World  # noqa: E402

SEV = {"info": "·", "warning": "▲", "critical": "■"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", nargs="?", default="demo_day")
    ap.add_argument("--all", action="store_true", help="include updates/resolutions")
    ap.add_argument("--json", help="write all events as JSON lines")
    args = ap.parse_args()
    world = World(get_scenario(args.scenario))
    pipe = DetectionPipeline(world.plans, write_incidents=False)
    t0 = time.time()
    while not world.done:
        frames = world.step()
        for ev in pipe.ingest(frames, world.site_snapshot(), world.t):
            if args.all or ev["status"] in ("open", "escalated"):
                print(f"{ev['ts'][11:16]} {SEV[ev['severity']]} {ev['severity'][:4]:4s} {ev['status'][:4]:4s} "
                      f"{ev['machine_id']:6s} {ev['intelligence_level'][:2]} {ev['type']:28s} {ev['summary'][:150]}")
    print(f"\n{len(pipe.events)} events in {time.time() - t0:.1f}s")
    if args.json:
        Path(args.json).write_text("\n".join(json.dumps(e) for e in pipe.events))


if __name__ == "__main__":
    main()
