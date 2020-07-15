import argparse

from ffn.utils import datspec as dx
from ffn.utils import unsupervised_metrics


ap = argparse.ArgumentParser()

ap.add_argument("datspecs", nargs="+")

args = ap.parse_args()

for datspec in args.datspecs:
    print(datspec)
    seg = dx.loadspec(datspec)
    metrics = unsupervised_metrics.unsupervised_metrics(seg)
    for name, met in zip(metrics._fields, metrics):
        print(f" - {name}: metric")
