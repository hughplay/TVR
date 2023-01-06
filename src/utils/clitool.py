import os
import subprocess


def execute(
    command,
    verbose: bool = False,
    silent: bool = False,
    bash: bool = False,
    timeout: float = None,
    log_path: str = None,
):
    """
    Note
    ----------
    This function is not recommended anymore. Use `sh` instead.

    References: https://amoffat.github.io/sh/
    """
    if bash:
        command = f'bash -c "{command}"'
    if verbose:
        print(command)
        print()
    try:
        if log_path is not None:
            out = open(log_path, "a")
        elif silent:
            out = open(os.devnull, "w")
        else:
            out = None
        p = subprocess.Popen(command, shell=True, stdout=out, stderr=out)
        if timeout is not None:
            print(f"Waiting {timeout}s to kill.")
        p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        raise  # resume the TimeoutExpired exception
    except KeyboardInterrupt:
        p.kill()
        raise  # resume the KeyboardInterrupt
    return p


def cuda_visible_devices_wrapper(command, device_ids=[]):
    """Use cuda_visible_devices to run a comand.

    Parameters
    ----------
    command : str
        The command to be executed.
    device_ids : list, optional
        The device ids to be used, by default [].

    Returns
    -------
    str
        The final command to be executed.
    """
    if len(device_ids) == 0:
        return command
    device_ids = ",".join(map(str, device_ids))
    command = f"CUDA_VISIBLE_DEVICES={device_ids} {command}"
    return command
