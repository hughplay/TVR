import argparse
import subprocess
import sys
import time
from multiprocessing import Process, Queue
from pathlib import Path

import torch

sys.path.append(".")
from src.utils.timetool import time2str  # noqa: E402


def get_commands(path_to_script):
    commands = []
    with open(path_to_script, "r") as f:
        lines = [
            line.strip() for line in f.readlines() if not line.startswith("#")
        ]
    command = ""
    for line in lines:
        if line.endswith("\\"):
            command += f"{line[:-1].strip()} "
        else:
            command += line
            if len(command.strip()) > 0:
                commands.append(command.strip())
            command = ""
    return commands


def run_command(command, gpus, logdir):
    gpu = None
    p = None
    try:
        t_start = time.time()
        while gpus.empty():
            time.sleep(1)
        gpu = gpus.get()
        name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        args = command.split()
        for arg in args:
            if arg.startswith("name="):
                name += f"-{arg.split('=')[1]}"
        log_path = Path(logdir) / f"{name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = f"CUDA_VISIBLE_DEVICES={gpu} {command} >> {log_path}"
        print(f"Running command: {command}")
        # command = "sleep infinity"
        p = subprocess.Popen(command, shell=True)
        p.wait()
        time.sleep(10)
    except KeyboardInterrupt:
        if p is not None:
            p.kill()
        exit(0)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if gpu is not None:
            gpus.put(gpu)
        print(
            f"Process {name} finished. "
            f"Time elapsed: {time2str(time.time() - t_start)}"
        )
        print(f"GPUs left: {gpus.qsize()}")


def main(commands, gpus, logdir):
    gpus_queue = Queue()
    for gpu in gpus:
        gpus_queue.put(gpu)

    processes = []
    for command in commands:
        process = Process(
            target=run_command, args=(command, gpus_queue, logdir)
        )
        time.sleep(1)
        process.start()
        processes.append(process)

    try:
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        for process in processes:
            print("Keyboard interrupt. Exiting.")
            if process.is_alive():
                print(f"Killing process {process.pid}")
                process.terminate()

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "script",
        type=str,
        help="Path to script to be run. Each line is a command.",
    )
    parser.add_argument("--gpus", type=str, default=None, help="GPUs to use")
    parser.add_argument(
        "--logdir", type=str, default="/log/running", help="Log directory"
    )
    args = parser.parse_args()

    if args.gpus is None:
        args.gpus = ",".join([str(i) for i in range(torch.cuda.device_count())])
    commands = get_commands(args.script)

    main(commands, args.gpus.split(","), args.logdir)
