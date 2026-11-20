"""Clinical-imaging and joint models.

The reported models are logistic regressions on an ascending-aortic diameter
B-spline, log-transformed root calcium volume, age and sex; the joint model adds
the logit of the deep-learning probability. Coefficients and knots are those in
params/joint_model_coefficients.json (Supplementary Table 4).

Example::

    python -m bavct.models.joint --input features.csv --out scores.csv --threshold 0.2385
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
PARAMS = REPO_ROOT / "params" / "joint_model_coefficients.json"


def bspline_basis(diameter, knots, degree=3):
    """Cox-de Boor basis for a clamped cubic B-spline.

    Returns the four basis functions that enter the model (N1..N4; N0 is the
    reference level and is dropped, as in the reported fit).
    """
    x = np.clip(np.asarray(diameter, dtype=float), knots[0], knots[-1])
    n = len(knots) - 1
    basis = np.zeros((n, x.size))
    for i in range(n):
        basis[i] = ((x >= knots[i]) & (x < knots[i + 1])).astype(float)
    basis[-1][x >= knots[-2]] = 1.0
    for k in range(1, degree + 1):
        nxt = np.zeros((n - k, x.size))
        for i in range(n - k):
            left = ((x - knots[i]) / (knots[i + k] - knots[i])
                    if knots[i + k] > knots[i] else 0.0)
            right = ((knots[i + k + 1] - x) / (knots[i + k + 1] - knots[i + 1])
                     if knots[i + k + 1] > knots[i + 1] else 0.0)
            nxt[i] = left * basis[i] + right * basis[i + 1]
        basis = nxt
    return basis[1:]


def score(diameter, calcium_mm3, age, male, dl_prob=None, model="joint",
          params_path=PARAMS):
    """Logistic-model probability for one or more examinations."""
    params = json.loads(Path(params_path).read_text(encoding="utf-8"))
    coef = params[{"clinical": "clinical_model", "joint": "joint_model"}[model]]
    knots = params["spline"]["knots"]
    basis = bspline_basis(diameter, knots)

    eta = np.full(basis.shape[1], float(coef["intercept"]))
    for i in range(4):
        eta += float(coef["B" + str(i + 1)]) * basis[i]
    eta += float(coef["logcalc"]) * np.log1p(np.asarray(calcium_mm3, float))
    eta += float(coef["age"]) * np.asarray(age, float)
    eta += float(coef["male"]) * np.asarray(male, float)
    if model == "joint":
        if dl_prob is None:
            raise ValueError("the joint model needs the DL probability")
        p = np.clip(np.asarray(dl_prob, float), 1e-6, 1 - 1e-6)
        eta += float(coef["logit_dl"]) * np.log(p / (1 - p))
    return 1.0 / (1.0 + np.exp(-eta))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", choices=["clinical", "joint"], default="joint")
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.input, newline="", encoding="utf-8-sig")))
    prob = score([float(r["diameter_mm"]) for r in rows],
                 [float(r["calcium_mm3"]) for r in rows],
                 [float(r["age"]) for r in rows],
                 [int(float(r["male"])) for r in rows],
                 [float(r["dl_prob"]) for r in rows] if args.model == "joint" else None,
                 model=args.model)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["exam_no", "prob", "positive"] if args.threshold is not None
                   else ["exam_no", "prob"])
        for r, p in zip(rows, prob):
            line = [r.get("exam_no", ""), "%.6f" % p]
            if args.threshold is not None:
                line.append(int(p >= args.threshold))
            w.writerow(line)


if __name__ == "__main__":
    main()
