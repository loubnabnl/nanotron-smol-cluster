# Submitting multiple slurm jobs
- `submit_jobs.py` can be used to train on some GPUs in a node in non exclusive mode.

- `submit_jobs_nodes.py` is similar to train on entire or multiple nodes (runs in exclusive mode, with multithreading).

```
# train models from index 0 to 40 from the spreadsheet, indexes are inclusive
python submit_jobs.py --start 0 --end 40
```
### Some useful commands:
This saves many logs, to find latest modified log file of the job at index 39 in the spreadsheet, for example, do:
```
log_path=$(find logs -name "*idx_39-*" -printf '%T@ %f\n' | sort -rn | head -n 1 | cut -d ' ' -f 2-)
tail -f logs/$log_path
```

To show full names of jobs (they are long and get truncated with `squeue`):
```
squeue --format="%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R" --me
```

To cancel all jobs from 2600 to 2640:
```
scancel $(seq 2675 2742)
```
