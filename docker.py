"""Run experiments in the docker container. Quick Start:

# Step 1. Install docker-compose in your workspace.
pip install docker-compose
# Step 2. Build docker image and start docker container once.
python docker.py startd --build
# Step 3. Enter the docker container at any time, start experiments now.
python docker.py

# Enter the docker container using root account.
python docker.py --root
"""
import argparse
import os
import signal
import subprocess
from collections import OrderedDict
from pathlib import Path

DEFAULT_ENV_PATH = ".env"
DEFAULT_PROJECT_NAME = "mnist"
DEFAULT_CODE_ROOT = "."
DEFAULT_DATA_ROOT = "data"
DEFAULT_LOG_ROOT = "log"
DEFAULT_CONTAINER_HOME = "./docker/misc/container_home"


class Env:
    def __init__(self, path=".env"):
        self.path = Path(path)
        self.lines, self.variables = self._parse_env()

    def _parse_env(self):
        lines = []
        variables = OrderedDict()
        if self.path.is_file():
            with open(self.path) as f:
                for i, line in enumerate(f):
                    lines.append(line)
                    line = line.strip()
                    if line and not line.startswith("#"):
                        key, val = line.split("=")
                    variables[key] = {"line": i, "key": key, "val": val}
        return lines, variables

    def __repr__(self):
        _strs = []
        for key, val in self.variables.items():
            _strs.append(f"  {key}: {val['val']}")
        return "\n".join(_strs)

    def __getitem__(self, key):
        return self.variables[key]["val"]

    def __setitem__(self, key, val):
        if key in self.variables:
            self.variables[key]["val"] = val
        else:
            self.variables[key] = {
                "line": len(self.lines),
                "key": key,
                "val": val,
            }
        self._update_line(self.variables[key])

    def __contains__(self, key):
        return key in self.variables

    def _update_line(self, var):
        line = f"{var['key']}={var['val']}\n"
        if var["line"] < len(self.lines):
            self.lines[var["line"]] = line
        else:
            self.lines.append(line)

    def save(self):
        with open(self.path, "w") as f:
            f.writelines(self.lines)


def execute(command):
    p = subprocess.Popen(command, shell=True)
    try:
        p.wait()
    except KeyboardInterrupt:
        try:
            os.kill(p.pid, signal.SIGINT)
        except OSError:
            pass
        p.wait()


def main(args):
    _set_env(verbose=(args.action == "start" or args.action == "startd"))

    SHELL = "zsh" if args.service == "project" else "bash"

    if args.action == "start" or args.action == "startd":
        command = "docker-compose up"
        if args.action == "startd":
            command += " -d"
        if args.build:
            command += " --build --force-recreate"
        command += f" {args.service}"
    elif args.action == "enter":
        if args.root:
            command = f"docker-compose exec -u root {args.service} {SHELL}"
        else:
            command = f"docker-compose exec {args.service} {SHELL}"
    else:
        command = f"docker-compose {args.action}"
    print(f"> {command}\n")
    execute(command)


def _set_env(env_path=DEFAULT_ENV_PATH, verbose=False):
    e = Env(env_path)
    e["UID"] = os.getuid()
    e["GID"] = os.getgid()
    e["USER_NAME"] = os.getlogin()

    def _get_value_from_stdin(prompt, default=None):
        value = input(f"{prompt} [{default}]: ").strip() or default
        return str(value)

    if "PROJECT" not in e:
        e["PROJECT"] = _get_value_from_stdin(
            "Give a project name", default=DEFAULT_PROJECT_NAME
        )

    if "CODE_ROOT" not in e:
        e["CODE_ROOT"] = _get_value_from_stdin(
            "Code root to be mounted at /project", default=DEFAULT_CODE_ROOT
        )

    if "DATA_ROOT" not in e:
        data_root = Path(
            _get_value_from_stdin(
                "Data root to be mounted at /data", default=DEFAULT_DATA_ROOT
            )
        ).resolve()
        if not data_root.exists():
            if (
                _get_value_from_stdin(
                    f"`{data_root}` does not exist in your machine. Create?",
                    default="yes",
                )
                == "yes"
            ):
                data_root.mkdir(parents=True)
        e["DATA_ROOT"] = str(data_root)

    if "LOG_ROOT" not in e:
        log_root = Path(
            _get_value_from_stdin(
                "Log root to be mounted at /log", default=DEFAULT_LOG_ROOT
            )
        ).resolve()
        if not log_root.exists():
            if (
                _get_value_from_stdin(
                    f"`{log_root}` does not exist in your machine. Create?",
                    default="yes",
                )
                == "yes"
            ):
                log_root.mkdir(parents=True)
        e["LOG_ROOT"] = str(log_root)

    if "CONTAINER_HOME" not in e:
        container_home = Path(
            _get_value_from_stdin(
                f"directory to be mounted to {e['USER_NAME']}",
                default=DEFAULT_CONTAINER_HOME,
            )
        ).resolve()
        if not container_home.exists():
            if (
                _get_value_from_stdin(
                    f"`{container_home}` does not exist in your machine. Create?",
                    default="yes",
                )
                == "yes"
            ):
                container_home.mkdir(parents=True, exist_ok=True)
        e["CONTAINER_HOME"] = str(container_home)

    e["COMPOSE_PROJECT_NAME"] = f"{e['PROJECT']}_{e['USER_NAME']}".lower()
    e.save()

    if verbose:
        print(f"Your setting ({env_path}):\n{e}\n")
    return e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="The core script of experiment management."
    )
    parser.add_argument("action", nargs="?", default="enter")
    parser.add_argument("-b", "--build", action="store_true", default=False)
    parser.add_argument("--root", action="store_true", default=False)
    parser.add_argument("--service", default="project")

    args = parser.parse_args()

    main(args)
