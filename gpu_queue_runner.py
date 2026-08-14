#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import time
import logging
import fcntl
import random

def setup_logging(log_file):
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def get_free_gpus(ignored_pattern="gmx", max_memory_mb=2048, max_gpu_util=20):
    # 1. Query all GPUs to map UUID to index and check memory/utilization thresholds
    try:
        res_gpus = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
    except Exception as e:
        logging.error(f"Failed to query GPUs list using nvidia-smi: {e}")
        return []

    gpu_list = []
    uuid_to_index = {}
    for line in res_gpus.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            idx, uuid, mem_used_str, util_str = parts[0], parts[1], parts[2], parts[3]
            try:
                mem_used = float(mem_used_str)
                gpu_util = float(util_str)
            except ValueError:
                logging.warning(f"Could not parse memory ({mem_used_str}) or util ({util_str}) for GPU {idx}. Skipping.")
                continue

            # First priority check: memory usage < 2GB (2048MB) and gpu util < 20%
            if mem_used >= max_memory_mb or gpu_util >= max_gpu_util:
                logging.debug(f"GPU {idx} filtered out: memory={mem_used}MB (max {max_memory_mb}MB), util={gpu_util}% (max {max_gpu_util}%)")
                continue

            gpu_list.append(idx)
            uuid_to_index[uuid] = idx

    if not gpu_list:
        return []

    # 2. Query compute processes for candidate GPUs
    gpu_processes = {idx: [] for idx in gpu_list}
    try:
        res_apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        for line in res_apps.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                uuid, pid, proc_name = parts[0], parts[1], parts[2]
                idx = uuid_to_index.get(uuid)
                if idx is not None:
                    gpu_processes[idx].append({
                        "pid": pid,
                        "process_name": proc_name
                    })
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.strip()
        if "no processes" in err_msg.lower() or "no active" in err_msg.lower() or not err_msg:
            pass  # assume no processes are running
        else:
            logging.warning(f"nvidia-smi --query-compute-apps returned error (code {e.returncode}): {err_msg}")
            return []
    except Exception as e:
        logging.error(f"Unexpected error querying compute apps: {e}")
        return []

    # 3. Filter free GPUs based on active processes and ignored patterns
    free_gpus = []
    for idx in gpu_list:
        procs = gpu_processes[idx]
        if not procs:
            free_gpus.append(idx)
        else:
            all_ignored = True
            for proc in procs:
                if ignored_pattern.lower() not in proc["process_name"].lower():
                    all_ignored = False
                    break
            if all_ignored:
                free_gpus.append(idx)

    return free_gpus

def get_and_pop_command(commands_file):
    if not os.path.exists(commands_file):
        return None

    try:
        with open(commands_file, "r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                lines = f.readlines()
                cmd_idx = -1
                cmd_to_run = None
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    if stripped and not stripped.startswith("#"):
                        cmd_to_run = stripped
                        cmd_idx = i
                        break
                
                if cmd_to_run is not None:
                    lines.pop(cmd_idx)
                    f.seek(0)
                    f.writelines(lines)
                    f.truncate()
                    return cmd_to_run
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except Exception as e:
        logging.error(f"Failed to pop command from queue file: {e}")
    return None

def append_past_command(command, past_commands_file):
    if not command:
        return
    try:
        with open(past_commands_file, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(command.strip() + "\n")
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except Exception as e:
        logging.error(f"Failed to append command to past commands file ({past_commands_file}): {e}")

def wrap_command_nohup(command, logs_dir, chosen_gpu):
    cmd_clean = command.strip()
    
    # 1. Remove trailing '&' if present
    if cmd_clean.endswith("&"):
        cmd_clean = cmd_clean[:-1].strip()
        
    # 2. Extract molecule name for descriptive log naming
    mol_name = "job"
    parts = cmd_clean.split()
    for i, part in enumerate(parts):
        if part == "--mol" and i + 1 < len(parts):
            mol_name = parts[i+1].strip("'\"")
            break
            
    # 3. Generate unique log path
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    rand_part = random.randint(100, 999)
    log_filename = f"{mol_name}_{timestamp}_{rand_part}_gpu{chosen_gpu}.log"
    log_filepath = os.path.join(logs_dir, log_filename)
    
    # 4. Check if command already starts with nohup. If not, prepend it.
    # Also check if it already has redirection. If it has redirection, we don't append our own redirect.
    has_redirect = ">" in cmd_clean
    
    if not cmd_clean.startswith("nohup"):
        cmd_clean = f"nohup {cmd_clean}"
        
    if not has_redirect:
        wrapped = f"{cmd_clean} > {log_filepath} 2>&1 &"
    else:
        wrapped = f"{cmd_clean} &"
        
    return wrapped, log_filepath

def main():
    parser = argparse.ArgumentParser(description="GPU Command Queue Runner")
    parser.add_argument("--root-dir", type=str, default="/ds/students/go38zin_leon_hamm/idp/CG-Mapping-Benchmark",
                        help="Root directory to run commands from")
    parser.add_argument("--commands-file", type=str, default="commands.txt",
                        help="Path to the file containing command queue lines")
    parser.add_argument("--past-commands-file", type=str, default="past_commands.txt",
                        help="Path to the file where popped/executed commands are appended")
    parser.add_argument("--interval", type=int, default=60,
                        help="Check interval in seconds when no GPUs are free or queue is empty")
    parser.add_argument("--wait-after-start", type=int, default=60,
                        help="Wait time in seconds after launching a script before checking again")
    parser.add_argument("--log-file", type=str, default="gpu_queue_runner.log",
                        help="Path to the runner log file")
    parser.add_argument("--ignored-process", type=str, default="gmx",
                        help="Process name keyword to ignore on GPUs")
    parser.add_argument("--max-memory-mb", type=float, default=2048,
                        help="Maximum allowed GPU memory usage in MB to consider GPU free (default: 2048)")
    parser.add_argument("--max-gpu-util", type=float, default=20,
                        help="Maximum allowed GPU utilization percentage to consider GPU free (default: 20)")
    parser.add_argument("--keep-alive", action="store_true",
                        help="Keep the runner running even when the queue is empty")
    
    args = parser.parse_args()
    
    root_dir = os.path.abspath(os.path.expanduser(args.root_dir))
    log_file_path = os.path.abspath(os.path.expanduser(args.log_file))
    commands_file_path = os.path.abspath(os.path.expanduser(args.commands_file))
    past_commands_file_path = os.path.abspath(os.path.expanduser(args.past_commands_file))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(script_dir, "runner_logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    setup_logging(log_file_path)
    
    logging.info("Starting GPU Queue Runner...")
    logging.info(f"Config: root_dir={root_dir}, commands_file={commands_file_path}, past_commands_file={past_commands_file_path}, interval={args.interval}s, wait_after_start={args.wait_after_start}s, ignored_process='{args.ignored_process}', max_memory_mb={args.max_memory_mb}, max_gpu_util={args.max_gpu_util}%, keep_alive={args.keep_alive}")
    
    if not os.path.isdir(root_dir):
        logging.error(f"Root directory does not exist: {root_dir}")
        sys.exit(1)
        
    while True:
        has_commands = False
        if os.path.exists(commands_file_path):
            try:
                with open(commands_file_path, "r") as f:
                    for line in f:
                        stripped = line.strip()
                        if stripped and not stripped.startswith("#"):
                            has_commands = True
                            break
            except Exception as e:
                logging.error(f"Error checking commands file: {e}")
                
        if not has_commands:
            if not args.keep_alive:
                logging.info("Queue is empty and --keep-alive is not set. Exiting.")
                sys.exit(0)
            logging.info("Queue is empty. Waiting...")
            time.sleep(args.interval)
            continue
            
        free_gpus = get_free_gpus(args.ignored_process, max_memory_mb=args.max_memory_mb, max_gpu_util=args.max_gpu_util)
        
        if not free_gpus:
            logging.info("No free GPUs available. Waiting...")
            time.sleep(args.interval)
            continue
            
        chosen_gpu = free_gpus[0]
        
        command = get_and_pop_command(commands_file_path)
        if not command:
            logging.info("Could not fetch command from queue (might have been removed). Retrying...")
            continue
            
        append_past_command(command, past_commands_file_path)
        logging.info(f"Found free GPU(s): {free_gpus}. Selected GPU {chosen_gpu}.")
        
        wrapped_command, log_filepath = wrap_command_nohup(command, logs_dir, chosen_gpu)
        logging.info(f"Launching wrapped command on GPU {chosen_gpu}: {wrapped_command}")
        
        env = os.environ.copy()
        env["GPU_CHOICE"] = str(chosen_gpu)
        env["CUDA_VISIBLE_DEVICES"] = str(chosen_gpu)
        
        try:
            subprocess.Popen(
                wrapped_command,
                shell=True,
                env=env,
                cwd=root_dir
            )
            logging.info(f"Command launched successfully. Output redirected to {log_filepath}. Sleeping for {args.wait_after_start} seconds to let it initialize.")
            time.sleep(args.wait_after_start)
        except Exception as e:
            logging.error(f"Failed to launch command: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
