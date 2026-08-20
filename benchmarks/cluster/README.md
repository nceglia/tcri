# Real-data profiling

`prep_real.py` and `profile_real.py` take every path, key and size as arguments — they run
anywhere. There is deliberately **no launcher script here.**

A slurm launcher has to name a filesystem root, a partition, an account and the datasets, all
of which are specific to one site and one person. Committing one to a public repository
publishes that site's layout and someone's account identifier, and it is useless to everybody
else. Keep yours in `dev/`, which is gitignored.

    python benchmarks/cluster/prep_real.py --in RAW.h5ad --out PREPPED.h5ad \
        --n-top-genes 2000 --clonotype-key CLONE_COL --target-col CLONE_COL --min-clone-size 10

    python benchmarks/cluster/profile_real.py --in PREPPED.h5ad --out RESULTS.json ...
