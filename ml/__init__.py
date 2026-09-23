import os

# Multithreaded OpenMP is pathologically slow for our small models on some machines
# (~170x on the dev laptop). Single-threaded is plenty; must be set before sklearn loads.
os.environ.setdefault("OMP_NUM_THREADS", "1")
