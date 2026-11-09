# N-Body Distributed Computing CCA

See **START_HERE.md** for the one-window GUI, precise timing methodology,
WSL launch instructions, reproducible benchmark commands and test checklist.

Run: `python integrated_gui.py` (or `bash run_demo.sh`).

Sequential and MPI compute equal fixed-step workloads from the same seed and
physics constants; the MPI option uses real `mpirun` ranks with mpi4py.
