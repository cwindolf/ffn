import argparse
from multiprocessing.pool import ThreadPool

from ffn.utils import datspec as dx
from ffn.utils import unsupervised_metrics


ap = argparse.ArgumentParser()

ap.add_argument("datspecs", nargs="+")

args = ap.parse_args()


def stats_thread(datspec):
    seg = dx.loadspec(datspec)
    metrics = unsupervised_metrics.unsupervised_metrics(seg)
    return datspec, metrics


with ThreadPool() as pool:
    for datspec, metrics in pool.imap(stats_thread, args.datspecs):
        print(datspec)
        for name, met in zip(metrics._fields, metrics):
            print(f" - {name}: {met}")
