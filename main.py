import sys
import time
import os
import json
from pathlib import Path
from typing import Optional
from src.config import MODELS_DIR, LOGS_DIR
from src.models.downloader import download_model
from src.models.manager import ModelManager


def get_device_temperature(device: str) -> Optional[float]:
    device_lower = device.lower()

    # 1. Try NVML/nvidia-smi if device is cuda/gpu
    if "cuda" in device_lower or "gpu" in device_lower:
        try:
            import subprocess
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=1
            )
            if res.returncode == 0 and res.stdout.strip():
                return float(res.stdout.strip())
        except Exception:
            pass
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            temp = pynvml.nvmlDeviceGetTemperature(handle, 0)
            pynvml.nvmlShutdown()
            return float(temp)
        except Exception:
            pass

    # 2. Try psutil sensors_temperatures for CPU (or GPU if listed there)
    try:
        import psutil
        if hasattr(psutil, "sensors_temperatures"):
            temps = psutil.sensors_temperatures()
            if temps:
                for key in ["coretemp", "cpu_thermal", "k10temp", "acpitz", "cpu-thermal"]:
                    if key in temps and temps[key]:
                        return float(temps[key][0].current)
                for name, entries in temps.items():
                    if entries:
                        return float(entries[0].current)
    except Exception:
        pass

    # 3. Linux sysfs CPU fallback
    if sys.platform.startswith("linux"):
        for zone in ["/sys/class/thermal/thermal_zone0/temp", "/sys/class/thermal/thermal_zone1/temp"]:
            try:
                if os.path.exists(zone):
                    with open(zone, "r") as f:
                        return float(f.read().strip()) / 1000.0
            except Exception:
                pass
        try:
            import glob
            for path in glob.glob("/sys/class/hwmon/hwmon*/temp*_input"):
                with open(path, "r") as f:
                    return float(f.read().strip()) / 1000.0
        except Exception:
            pass

    # 4. Windows PowerShell CPU fallback (using non-admin Perf counters first, then admin MSAcpi fallback)
    if sys.platform == "win32":
        try:
            import subprocess
            cmd = "Get-CimInstance -ClassName Win32_PerfFormattedData_Counters_ThermalZoneInformation | Where-Object { $_.Name -like '*CPU*' } | Select-Object -ExpandProperty HighPrecisionTemperature"
            res = subprocess.run(
                ["powershell", "-Command", cmd],
                capture_output=True,
                text=True,
                timeout=1
            )
            val = res.stdout.strip()
            if val:
                return (float(val.split()[0]) / 10.0) - 273.15
        except Exception:
            pass

        try:
            import subprocess
            cmd = "Get-CimInstance -ClassName Win32_PerfFormattedData_Counters_ThermalZoneInformation | Where-Object { $_.Name -like '*CPU*' } | Select-Object -ExpandProperty Temperature"
            res = subprocess.run(
                ["powershell", "-Command", cmd],
                capture_output=True,
                text=True,
                timeout=1
            )
            val = res.stdout.strip()
            if val:
                return float(val.split()[0]) - 273.15
        except Exception:
            pass

        try:
            import subprocess
            cmd = "Get-CimInstance -ClassName Win32_PerfFormattedData_Counters_ThermalZoneInformation | Select-Object -ExpandProperty HighPrecisionTemperature"
            res = subprocess.run(
                ["powershell", "-Command", cmd],
                capture_output=True,
                text=True,
                timeout=1
            )
            val = res.stdout.strip()
            if val:
                temps = [float(v) for v in val.split() if float(v) > 0]
                if temps:
                    return (temps[0] / 10.0) - 273.15
        except Exception:
            pass

        try:
            import subprocess
            cmd = "Get-CimInstance -Namespace root/wmi -ClassName MSAcpi_ThermalZoneTemperature | Select-Object -ExpandProperty CurrentTemperature"
            res = subprocess.run(
                ["powershell", "-Command", cmd],
                capture_output=True,
                text=True,
                timeout=1
            )
            if res.returncode == 0 and res.stdout.strip():
                return (float(res.stdout.strip()) / 10.0) - 273.15
        except Exception:
            pass

    return None


def get_device_utilization(device: str) -> Optional[float]:
    """return device utilization as a percentage, or None if unavailable."""
    device_lower = device.lower()
    if "cuda" in device_lower or "gpu" in device_lower:
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            pynvml.nvmlShutdown()
            return float(util.gpu)
        except Exception:
            pass
    try:
        import psutil
        return psutil.cpu_percent(interval=None)
    except Exception:
        return None


def get_ram_usage_mb() -> Optional[float]:
    """return current process ram usage in mb, or None if unavailable."""
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def get_system_ram_mb() -> Optional[dict]:
    """return system-wide ram stats in mb."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {
            "total": vm.total / (1024 * 1024),
            "used": vm.used / (1024 * 1024),
            "available": vm.available / (1024 * 1024),
            "percent": vm.percent,
        }
    except Exception:
        return None


import threading

class TemperatureTracker(threading.Thread):
    def __init__(self, device: str, interval: float = 1.0) -> None:
        super().__init__(daemon=True)
        self.device = device
        self.interval = interval
        self.temperatures = []
        self._stop_event = threading.Event()

    def run(self) -> None:
        temp = get_device_temperature(self.device)
        if temp is not None:
            self.temperatures.append(temp)
        while not self._stop_event.is_set():
            if self._stop_event.wait(self.interval):
                break
            temp = get_device_temperature(self.device)
            if temp is not None:
                self.temperatures.append(temp)

    def stop(self) -> None:
        self._stop_event.set()


def parse_run_args(args: list) -> dict:
    """
    parse flags for the run command.
    returns a dict with: model_name, prompt, verbose, silent, max_tokens, temperature, log, log_file
    """
    opts = {
        "model_name": None,
        "prompt": "",
        "verbose": False,
        "silent": False,
        "bare": False,
        "max_tokens": 256,
        "temperature": 0.7,
        "log": False,
        "log_file": None,
    }

    positional = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("--verbose", "-v"):
            opts["verbose"] = True
        elif arg in ("--silent", "-s"):
            opts["silent"] = True
        elif arg in ("--bare", "-b"):
            opts["bare"] = True
        elif arg == "--log":
            opts["log"] = True
        elif arg == "--log-file":
            i += 1
            if i < len(args):
                opts["log_file"] = args[i]
                opts["log"] = True
        elif arg == "--max-tokens":
            i += 1
            if i < len(args):
                try:
                    opts["max_tokens"] = int(args[i])
                except ValueError:
                    print(f"error: --max-tokens expects an integer, got '{args[i]}'")
                    sys.exit(1)
        elif arg == "--temperature":
            i += 1
            if i < len(args):
                try:
                    opts["temperature"] = float(args[i])
                except ValueError:
                    print(f"error: --temperature expects a float, got '{args[i]}'")
                    sys.exit(1)
        else:
            positional.append(arg)
        i += 1

    if len(positional) >= 1:
        opts["model_name"] = positional[0]
    if len(positional) >= 2:
        opts["prompt"] = positional[1]

    return opts


def main() -> None:
    """
    command line entrypoint for strata.
    supports:
    - uv run main.py download {link}
    - uv run main.py run {model_name} [prompt] [--verbose|-v] [--silent|-s] [--max-tokens N] [--temperature F] [--log] [--log-file path]
    - uv run main.py list models | logs
    - uv run main.py rm {model_name}
    - uv run main.py rename {old_model_name} {new_model_name}
    - uv run main.py view {filename | path}
    """
    args = sys.argv[1:]

    if len(args) < 1:
        print("usage: uv run main.py download {link} | run {model_name} [prompt] [--verbose] [--silent] [--max-tokens N] [--temperature F] [--log] [--log-file path] | list models | logs | rm {model_name} | rename {old} {new} | view {file}")
        sys.exit(1)

    command: str = args[0].lower()

    if command == "list":
        if len(args) < 2:
            print("usage: uv run main.py list models | logs")
            sys.exit(1)
        subcommand = args[1].lower()
        if subcommand == "models":
            manager = ModelManager()
            try:
                models = manager.get_available_models()
                if not models:
                    print("* no models downloaded yet")
                else:
                    for m in models:
                        print(f"* {m['name'].lower()} / {m['type'].lower()}")
            except Exception as err:
                print(f"error: {str(err).lower()}")
                sys.exit(1)
        elif subcommand == "logs":
            try:
                logs = sorted(LOGS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                if not logs:
                    print("* no logs yet")
                else:
                    for log in logs:
                        print(f"* {log.name}")
            except Exception as err:
                print(f"error: {str(err).lower()}")
                sys.exit(1)
        else:
            print(f"unknown list subcommand: {subcommand}")
            print("usage: uv run main.py list models | logs")
            sys.exit(1)
        sys.exit(0)

    elif command == "rm":
        if len(args) < 2:
            print("usage: uv run main.py rm {model_name}")
            sys.exit(1)
        model_name: str = args[1]
        manager = ModelManager()
        try:
            manager.delete_model(model_name)
            print(f"model '{model_name.lower()}' removed successfully")
        except Exception as err:
            print(f"error: {str(err).lower()}")
            sys.exit(1)
        sys.exit(0)

    elif command == "rename":
        if len(args) < 3:
            print("usage: uv run main.py rename {old_model_name} {new_model_name}")
            sys.exit(1)
        old_name: str = args[1]
        new_name: str = args[2]
        manager = ModelManager()
        try:
            manager.rename_model(old_name, new_name)
            print(f"model '{old_name.lower()}' renamed to '{new_name.lower()}' successfully")
        except Exception as err:
            print(f"error: {str(err).lower()}")
            sys.exit(1)
        sys.exit(0)

    elif command == "download":
        if len(args) < 2:
            print("usage: uv run main.py download {link}")
            sys.exit(1)
        link: str = args[1]
        print(f"starting download for link: {link.lower()}")
        try:
            download_model(
                link=link,
                dest_dir=MODELS_DIR,
                progress_callback=lambda msg: print(msg.lower())
            )
            print("download finished successfully")
        except Exception as err:
            print(f"error: {str(err).lower()}")
            sys.exit(1)

    elif command == "run":
        run_args = parse_run_args(args[1:])

        if run_args["model_name"] is None:
            print("usage: uv run main.py run {model_name} [prompt] [--verbose|-v] [--silent|-s] [--max-tokens N] [--temperature F] [--log] [--log-file path]")
            sys.exit(1)

        model_name: str = run_args["model_name"]
        prompt: str = run_args["prompt"]
        verbose: bool = run_args["verbose"]
        silent: bool = run_args["silent"]
        bare: bool = run_args["bare"]
        max_tokens: int = run_args["max_tokens"]
        temperature: float = run_args["temperature"]
        do_log: bool = run_args["log"]
        log_file: Optional[str] = run_args["log_file"]

        if verbose and silent:
            print("error: --verbose and --silent are mutually exclusive")
            sys.exit(1)

        manager = ModelManager()
        try:
            ram_before = get_ram_usage_mb() if not bare else None
            sys_ram_before = get_system_ram_mb() if not bare else None
            harness = manager.load_model(model_name)
            ram_after_load = get_ram_usage_mb() if not bare else None
            sys_ram_after_load = get_system_ram_mb() if not bare else None
            device = harness.device
            prompt_tokens = manager.count_tokens(prompt) if prompt else 0

            if verbose:
                print(f"* model / {model_name.lower()}")
                print(f"* device / {device.lower()}")
                print(f"* max_tokens / {max_tokens}")
                print(f"* temperature / {temperature}")
                if ram_before is not None and ram_after_load is not None:
                    print(f"* ram before load / {ram_before:.1f}mb")
                    print(f"* ram after load / {ram_after_load:.1f}mb")
                    print(f"* ram delta (load) / {ram_after_load - ram_before:.1f}mb")
                if sys_ram_after_load:
                    print(f"* system ram / {sys_ram_after_load['used']:.0f}mb used / {sys_ram_after_load['total']:.0f}mb total / {sys_ram_after_load['percent']:.1f}% used")
                if prompt:
                    print(f"* prompt tokens / {prompt_tokens}")
                else:
                    print("* prompt / (empty)")
                print()
            elif not silent:
                print(f"\n* strata / {device.lower()}\n")

            # start background temperature tracker (skipped in bare mode)
            tracker = TemperatureTracker(device, interval=1.0)
            if not bare:
                tracker.start()

            response_chunks = []
            chunk_times = []
            chunk_token_log = []
            total_tokens_so_far = 0
            start_time = time.time()
            last_time = start_time

            for chunk in manager.generate_response_stream(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature
            ):
                current_time = time.time()

                dt = current_time - last_time
                num_tokens = manager.count_tokens(chunk)
                if num_tokens == 0 and chunk:
                    num_tokens = 1

                total_tokens_so_far += num_tokens
                elapsed = current_time - start_time

                if silent:
                    # overwrite the current line with token count and elapsed time
                    print(f"\r* tokens / {total_tokens_so_far} / {elapsed:.1f}s", end="", flush=True)
                else:
                    print(chunk.lower(), end="", flush=True)

                if num_tokens > 0 and dt > 0:
                    tps = num_tokens / dt
                    chunk_times.append(tps)
                    chunk_ram = get_ram_usage_mb() if not bare else None
                    chunk_util = get_device_utilization(device) if not bare else None
                    entry = {
                        "chunk": chunk,
                        "tokens": num_tokens,
                        "tps": round(tps, 2),
                        "elapsed_s": round(elapsed, 3),
                    }
                    if not bare:
                        entry["ram_mb"] = round(chunk_ram, 2) if chunk_ram is not None else None
                        entry["device_util_pct"] = round(chunk_util, 1) if chunk_util is not None else None
                    chunk_token_log.append(entry)

                response_chunks.append(chunk)
                last_time = current_time

            if silent:
                print()  # newline after the counter line
            else:
                print()

            # stop background tracker, grab one final reading (skipped in bare mode)
            if not bare:
                tracker.stop()
                last_temp = get_device_temperature(device)
                if last_temp is not None:
                    tracker.temperatures.append(last_temp)
                tracker.join(timeout=0.5)
            chunk_temps = tracker.temperatures if not bare else []

            ram_after_inference = get_ram_usage_mb() if not bare else None
            sys_ram_final = get_system_ram_mb() if not bare else None

            time_took = time.time() - start_time
            full_response = "".join(response_chunks)
            token_count = manager.count_tokens(full_response)
            tokens_per_second = token_count / time_took if time_took > 0 else 0.0

            if chunk_times:
                peak_tps = max(chunk_times)
                lowest_tps = min(chunk_times)
            else:
                peak_tps = tokens_per_second
                lowest_tps = tokens_per_second

            if chunk_temps:
                avg_temp = sum(chunk_temps) / len(chunk_temps)
                peak_temp = max(chunk_temps)
                lowest_temp = min(chunk_temps)
                temp_str = f"{avg_temp:.1f}c overall / {peak_temp:.1f}c peak / {lowest_temp:.1f}c lowest"
            else:
                avg_temp = peak_temp = lowest_temp = None
                temp_str = "n/a"

            if not silent:
                print(f"\n* time / {time_took:.2f}s")
                print(f"* tokens / {token_count}")
                print(f"* tps / {tokens_per_second:.2f} overall / {peak_tps:.2f} peak / {lowest_tps:.2f} lowest")
                print(f"* temp / {temp_str}")
                if ram_after_load is not None and ram_after_inference is not None:
                    print(f"* ram / {ram_after_inference:.1f}mb process / {ram_after_inference - ram_after_load:.1f}mb delta (inference)")
                if sys_ram_final:
                    print(f"* system ram / {sys_ram_final['used']:.0f}mb used / {sys_ram_final['total']:.0f}mb total / {sys_ram_final['percent']:.1f}% used")
                print()

                if verbose and chunk_token_log:
                    print("* per-chunk token trace:")
                    for entry in chunk_token_log:
                        print(f"  [{entry['elapsed_s']:.3f}s] {entry['tokens']} tok @ {entry['tps']} tps")
                    print()

            # save json log if requested — always contains full data regardless of verbose/silent
            if do_log:
                stats = {
                    "time_s": round(time_took, 4),
                    "prompt_tokens": prompt_tokens,
                    "response_tokens": token_count,
                    "tps_overall": round(tokens_per_second, 4),
                    "tps_peak": round(peak_tps, 4),
                    "tps_lowest": round(lowest_tps, 4),
                }
                if not bare:
                    stats.update({
                        "temp_avg_c": round(avg_temp, 2) if avg_temp is not None else None,
                        "temp_peak_c": round(peak_temp, 2) if peak_temp is not None else None,
                        "temp_lowest_c": round(lowest_temp, 2) if lowest_temp is not None else None,
                        "ram_before_load_mb": round(ram_before, 2) if ram_before is not None else None,
                        "ram_after_load_mb": round(ram_after_load, 2) if ram_after_load is not None else None,
                        "ram_load_delta_mb": round(ram_after_load - ram_before, 2) if (ram_before is not None and ram_after_load is not None) else None,
                        "ram_after_inference_mb": round(ram_after_inference, 2) if ram_after_inference is not None else None,
                        "ram_inference_delta_mb": round(ram_after_inference - ram_after_load, 2) if (ram_after_inference is not None and ram_after_load is not None) else None,
                        "system_ram_before": sys_ram_before,
                        "system_ram_after_load": sys_ram_after_load,
                        "system_ram_after_inference": sys_ram_final,
                    })

                log_data = {
                    "model": model_name,
                    "device": device,
                    "prompt": prompt,
                    "response": full_response,
                    "args": {
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                    "stats": stats,
                    "chunk_trace": chunk_token_log,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }

                if log_file is None:
                    log_file = str(LOGS_DIR / f"strata_{model_name}_{int(start_time)}.json")

                with open(log_file, "w", encoding="utf-8") as f:
                    json.dump(log_data, f, indent=2)

                if not silent:
                    print(f"* log saved to {log_file}\n")

        except Exception as err:
            print(f"\nerror: {str(err).lower()}")
            sys.exit(1)

    elif command == "view":
        if len(args) < 2:
            print("usage: uv run main.py view {filename | path}")
            sys.exit(1)

        target = args[1]
        target_path = Path(target)

        # if not a path that exists, look in logs dir
        if not target_path.exists():
            candidate = LOGS_DIR / target
            if candidate.exists():
                target_path = candidate
            else:
                print(f"error: file not found: {target}")
                sys.exit(1)

        import subprocess

        if sys.platform == "win32":
            try:
                subprocess.run(["notepad.exe", str(target_path)])
            except Exception as err:
                print(f"error: could not open notepad: {str(err).lower()}")
                sys.exit(1)
        else:
            for editor in ("nano", "vim", "vi"):
                try:
                    result = subprocess.run(["which", editor], capture_output=True, text=True)
                    if result.returncode == 0:
                        subprocess.run([editor, str(target_path)])
                        break
                except Exception:
                    continue
            else:
                print("error: no suitable editor found (nano, vim, vi)")
                sys.exit(1)

    else:
        print(f"unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
