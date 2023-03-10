# Submitting multiple slurm jobs
- `submit_jobs.py` can be used to train on some GPUs in a node in non exclusive mode.

- `submit_jobs_nodes.py` is similar but runs in exclusive mode (with multithreading).

```
# train models from index 0 to 40 from the spreadsheet, indexes are inclusive
python submit_jobs.py --start 0 --end 40
```
