import os
import subprocess
from typing import List
from itertools import cycle
from shutil import get_terminal_size
from threading import Thread
from time import sleep


# change these paths and run `python llm_swarm_launch.py`
SLURM_LOGS_FOLDER = "/fsx/loubna/projects/llm-swarm/slurm/logs"
slurm_path = "/fsx/loubna/projects/llm-swarm/slurm/tgi_1711535789_tgi.slurm"
number_tgi_instances = 10

def run_command(command: str):
    print(f"running {command}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, errors = process.communicate()
    return_code = process.returncode
    assert return_code == 0, f"Command failed with error: {errors.decode('utf-8')}"
    return output.decode("utf-8").strip()


def is_job_running(job_id: str):
    """Given job id, check if the job is in eunning state (needed to retrieve hostname from logs)"""
    command = "squeue --me --states=R | awk '{print $1}' | tail -n +2"
    my_running_jobs = subprocess.run(command, shell=True, text=True, capture_output=True).stdout.splitlines()
    return job_id in my_running_jobs


class Loader:
    def __init__(self, desc="Loading...", end="✅ Done!", failed="❌ Aborted!", timeout=0.1):
        """
        A loader-like context manager
        Modified from https://stackoverflow.com/a/66558182/6611317

        Args:
            desc (str, optional): The loader's description. Defaults to "Loading...".
            end (str, optional): Final print. Defaults to "Done!".
            failed (str, optional): Final print on failure. Defaults to "Aborted!".
            timeout (float, optional): Sleep time between prints. Defaults to 0.1.
        """
        self.desc = desc
        self.end = end + " " + self.desc
        self.failed = failed + " " + self.desc
        self.timeout = timeout

        self._thread = Thread(target=self._animate, daemon=True)
        self.steps = ["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"]
        self.done = False

    def start(self):
        self._thread.start()
        return self

    def _animate(self):
        try:
            for c in cycle(self.steps):
                if self.done:
                    break
                print(f"\r{c} {self.desc}", flush=True, end="")
                sleep(self.timeout)
        except KeyboardInterrupt:
            self.stop()
            print("KeyboardInterrupt by user")

    def __enter__(self):
        self.start()

    def stop(self):
        self.done = True
        cols = get_terminal_size((80, 20)).columns
        print("\r" + " " * cols, end="", flush=True)
        print(f"\r{self.end}", flush=True)

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is None:
            self.stop()
        else:
            self.done = True
            cols = get_terminal_size((80, 20)).columns
            print("\r" + " " * cols, end="", flush=True)
            print(f"\r{self.failed}", flush=True)

job_ids = [run_command(f"sbatch --parsable {slurm_path}") for _ in range(number_tgi_instances)]
try:
    # ensure job is running
    for job_id in job_ids:
        with Loader(f"Waiting for {job_id} to be created"):
            while not is_job_running(job_id):
                sleep(1)
        slumr_log_path = os.path.join(SLURM_LOGS_FOLDER, f"llm-swarm_{job_id}.out")
        print(f"📖 Slurm log path: {slumr_log_path}")
except:
    print("Failed try...")
