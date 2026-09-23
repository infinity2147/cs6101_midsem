"""Unpack artifacts/ into WORK (data) and RUNS (per-model outputs) so analysis/significance run
without re-training. Also rewrites args.json paths to the new locations."""
import argparse, glob, gzip, json, os, shutil

ap = argparse.ArgumentParser()
ap.add_argument("--work", default="work/squad")
ap.add_argument("--runs", default="work/runs")
a = ap.parse_args()
root = os.path.join(os.path.dirname(__file__), "..", "artifacts")
os.makedirs(a.work, exist_ok=True)
for f in glob.glob(os.path.join(root, "data", "*.gz")):
    with gzip.open(f, "rb") as src, open(os.path.join(a.work, os.path.basename(f)[:-3]), "wb") as dst:
        shutil.copyfileobj(src, dst)
for d in glob.glob(os.path.join(root, "runs", "*")):
    out = os.path.join(a.runs, os.path.basename(d))
    os.makedirs(out, exist_ok=True)
    for f in glob.glob(os.path.join(d, "*")):
        name = os.path.basename(f)
        if name.endswith(".gz"):
            with gzip.open(f, "rb") as src, open(os.path.join(out, name[:-3]), "wb") as dst:
                shutil.copyfileobj(src, dst)
        else:
            shutil.copy(f, out)
    p = os.path.join(out, "args.json")
    args = json.load(open(p))
    args["data"] = os.path.abspath(a.work)
    args["triples"] = os.path.join(os.path.abspath(a.work), os.path.basename(args["triples"]))
    if args.get("init"):
        args["init"] = os.path.join(os.path.abspath(a.runs), os.path.basename(args["init"]))
    json.dump(args, open(p, "w"), indent=1)
print(f"restored data -> {a.work}, runs -> {a.runs}")
