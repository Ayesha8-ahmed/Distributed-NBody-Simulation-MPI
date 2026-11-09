# N-Body CCA — one integrated GUI (Sequential + MPI)

**Use `python integrated_gui.py`. There is no menu in the terminal.**

## Run in WSL Ubuntu

1. Copy the downloaded `NBody_CCA_Final_Simple_Integrated.zip` into `D:\UbuntuShare` on Windows.
2. Open your **WSL Ubuntu** terminal (not Windows PowerShell) and run:

```bash
cd ~
python3 -m zipfile -e /mnt/d/UbuntuShare/NBody_CCA_Final_Simple_Integrated.zip .
cd ~/NBody_CCA_Final_Simple_Integrated
source ~/NBody_Integrated_Ubuntu/.venv/bin/activate
python integrated_gui.py
```

If you prefer, the last command can be `bash run_demo.sh` (same GUI, no menu). If your earlier `.venv` does not exist, follow `bash setup_ubuntu.sh` inside this project before launching.

## GUI workflow

- Click **Sequential** or **MPI** at the top.
- Select **Bodies**: 10, 50, 100, 103, 500, 1000 or 5000.
- Select **Steps**: 10, 50, 100, 500 or 1000.
- In MPI mode choose **Processes**: 1, 2, 4 or 8. Process controls are disabled in Sequential mode.
- Click **Start** once. The application initializes the engine and then automatically executes **exactly the selected number of steps**. It stops at FINISHED, with final execution time frozen on screen.
- Click **Metrics**, **Timing**, **Workload**, or **Charts** to inspect results. Use **Stop** to cancel a long run; canceled runs are **not** included in comparisons.
- To compare: run Sequential for 100 bodies / 100 steps; switch to MPI with 2 processes, run again; switch to MPI with 4 processes, run again. Open **Charts**. The chart automatically compares completed runs with matching bodies and steps. You do not need a separate Compare button.

### What is a step?

One physics step: calculate the gravitational acceleration of **each** body using the current positions, then update all velocities and positions one time. A selection of 100 steps performs that cycle 100 times, **not 100 seconds**. `dt=0.02` is a simulated-time increment, not computer execution time.

## Timing: exact definition

The application uses `time.perf_counter()` once **before** the fixed-length physics loop and once **after** its last step:

```
initialize engine (not timed)
MPI ranks synchronize before timing (MPI only)
start = perf_counter()
repeat exactly S steps: compute forces, update motion, MPI exchange and synchronization
MPI ranks synchronize after all S steps (MPI only)
elapsed = perf_counter() - start
MPI uses maximum elapsed across ranks
```

**Total execution time** includes the full simulation calculation, Python loop overhead and MPI communication/synchronization. It **does not** include initial generation of bodies, MPI process startup, or the separate Pygame drawing loop. The GUI updates planet positions periodically; sending those preview snapshots adds some I/O overhead inside GUI runs, identical in intent for Sequential/MPI. Hence use the **no-preview repeated terminal benchmark** below for the most defensible report numbers. The timer shown live is an approximation; after FINISHED it is replaced by the actual measured start/end result.

**Computation time** and **communication time** are supporting diagnostics. They are sums of per-step maximum-rank measurements and are **not guaranteed to add up to total execution time**. Do not label the old standalone engine's last-step numbers as complete-run measurements.

## Fair repeated benchmarks for CCA report

The terminal benchmark uses the **same integrated_worker.py engine** but disables preview data and performs 3 separate runs per configuration:

```bash
cd ~/NBody_CCA_Final_Simple_Integrated
source ~/NBody_Integrated_Ubuntu/.venv/bin/activate
python fair_benchmark.py --bodies 100,500 --processes 1,2,4 --steps 100 --repeats 3
```

Measured results are saved to `benchmarking/fair_results.csv`. Console summaries show mean execution time, speedup (Sequential mean / MPI mean), and parallel efficiency (speedup / MPI ranks × 100%). MPI may be **slower** than Sequential for small N because of communication costs; that is a legitimate performance finding.

## Sanity checks before the teacher demo

```bash
python -m unittest -v test_main test_integrated_runner
mpirun -np 4 python verify_mpi.py 103 10
```

Expected: all sequential tests pass, the integration tests pass including a two-rank MPI test, and numerical verifier prints `RESULT: PASS`. To demonstrate uneven workload select **103 bodies / 4 processes**: ranks should own **26, 26, 26, 25** bodies.

The project's earlier standalone `main.py`, `mpi_gui.py`, `mpi_engine.py` are preserved. Historical benchmark files are under `legacy/` so they cannot be confused with the new timing results. Only `integrated_gui.py`, `integrated_worker.py` and `fair_benchmark.py` implement the new workflow.

## CCA evidence

The interface is for demonstration. For the **report**, also include performance and scalability runs across multiple body counts and MPI process counts (1, 2, 4, and 8 where feasible), numerical correctness tolerance, the machine specifications, computation and communication measurements, workload distribution, speedup, efficiency, graphs and conclusions. Repeat experiments and note GUI results and no-preview benchmark results have different instrumentation overhead.

**Validation note:** Source syntax, the Sequential unit tests, the Sequential worker end-to-end flow and a simulated one-rank MPI physics test were checked during packaging. The multi-rank MPI GUI could not be executed in the packaging environment (MPI/Pygame dependencies unavailable), so verify it in WSL using the commands above. Do not claim this project is fully runtime-certified before that test.
