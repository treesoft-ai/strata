import sys
import time
import os
import json
import logging
import warnings
from pathlib import Path
from typing import Optional

# Suppress python warnings and third-party library logging for TTI compliance
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from src.config import MODELS_DIR, LOGS_DIR, TRAINED_DIR
from src.models.downloader import download_model
from src.models.manager import ModelManager


def print_header(section: str) -> None:
    print(f"* Strata / {section}\n")


def print_error(section: str, message: str, next_action: str) -> None:
    print_header(section)
    print(f"  ! Error: {message}\n")
    print(f"  {next_action}")


def print_root_help() -> None:
    print_header("Help")
    print("  Commands\n")
    print("  download    Download a model from Hugging Face")
    print("  run         Run inference on a model")
    print("  list        List models or logs")
    print("  rm          Remove a model")
    print("  rename      Rename a model")
    print("  train       Fine-tune a model on a Strata JSONL dataset")
    print("  synthesize  Generate synthetic JSONL datasets with AI")
    print("  view        View a log file\n")
    print("  Run 'uv run main.py <command> --help' for details on a command.")


def print_download_help() -> None:
    print_header("Download")
    print("  Usage: uv run main.py download {link}\n")
    print("  Downloads a model from Hugging Face by URL or repository shorthand.")


def print_list_help() -> None:
    print_header("Models")
    print("  Usage: uv run main.py list models | logs\n")
    print("  Lists local downloaded models or execution logs.")


def print_rm_help() -> None:
    print_header("Models")
    print("  Usage: uv run main.py rm {model_name}\n")
    print("  Deletes a model local directory from persistent storage.")


def print_rename_help() -> None:
    print_header("Models")
    print("  Usage: uv run main.py rename {old_name} {new_name}\n")
    print("  Renames a model local directory in persistent storage.")


def print_view_help() -> None:
    print_header("Logs")
    print("  Usage: uv run main.py view {filename | path}\n")
    print("  Opens the specified log file in Notepad (Windows) or a CLI editor (Linux).")


def print_train_help() -> None:
    print_header("Train")
    print("  Usage: uv run main.py train {model_name} --data {file.jsonl} [options]\n")
    print("  Fine-tunes a local Transformers model using a Strata JSONL dataset.\n")
    print("  Required")
    print("  --data path          Path to a Strata .jsonl training dataset\n")
    print("  Method")
    print("  --method lora        LoRA fine-tune (default)")
    print("  --method qlora       QLoRA 4-bit fine-tune (low VRAM)")
    print("  --method full        Full parameter fine-tune")
    print("  --method dpo         DPO preference training")
    print("  --method grpo        GRPO reinforcement learning")
    print("  --method ppo         PPO reinforcement learning\n")
    print("  LoRA / QLoRA")
    print("  --lora-r N           LoRA rank (default: 16)")
    print("  --lora-alpha N       LoRA alpha (default: 32)")
    print("  --lora-dropout F     LoRA dropout (default: 0.05)")
    print("  --lora-modules m,..  Target modules, comma-separated (default: q_proj,v_proj)\n")
    print("  Training loop")
    print("  --epochs N           Number of training epochs (default: 3)")
    print("  --batch-size N       Per-device batch size (default: 2)")
    print("  --grad-accum N       Gradient accumulation steps (default: 4)")
    print("  --lr F               Learning rate (default: 2e-4)")
    print("  --warmup N           Warmup steps (default: 10)")
    print("  --max-seq N          Maximum sequence length (default: 512)")
    print("  --weight-decay F     Weight decay (default: 0.01)")
    print("  --seed N             Random seed (default: 42)\n")
    print("  DPO")
    print("  --dpo-beta F         KL penalty coefficient (default: 0.1)\n")
    print("  GRPO / PPO")
    print("  --grpo-beta F        GRPO KL penalty coefficient (default: 0.04)")
    print("  --ppo-beta F         PPO KL penalty coefficient (default: 0.1)")
    print("  --reward-model path  HF reward model path for PPO")
    print("  --reward-fn path     Python script with reward(prompt, response)->float")
    print("  --num-generations N  GRPO completions per prompt (default: 4)")
    print("  --max-new-tokens N   Max tokens to generate per RL step (default: 256)")
    print("  --use-lora           Apply LoRA adapters during GRPO/PPO\n")
    print("  Output")
    print("  --run-name name      Name for the training run (default: model_name + timestamp)")
    print("  --no-resume          Do not resume from an existing checkpoint\n")
    print("  Dataset format (Strata JSONL — one JSON object per line)")
    print("  SFT:   {\"prompt\": \"...\", \"completion\": \"...\"}")
    print("  Chat:  {\"messages\": [{\"role\": \"user\", \"content\": \"...\"}, ...]}")
    print("  DPO:   {\"prompt\": \"...\", \"chosen\": \"...\", \"rejected\": \"...\"}")
    print("  RL:    {\"prompt\": \"...\"}  (GRPO / PPO)")


def print_run_help() -> None:
    print_header("Run")
    print("  Usage: uv run main.py run {model_name} [prompt] [options]\n")
    print("  Runs local model inference and outputs hardware diagnostics.\n")
    print("  Options")
    print("  --verbose, -v    Show detailed diagnostics and per-chunk token trace")
    print("  --silent, -s     Show minimal token count and execution time progress")
    print("  --bare, -b       Disable diagnostic background temperature & RAM tracking")
    print("  --max-tokens N   Maximum tokens to generate (default: 256)")
    print("  --temperature F  Sampling temperature (default: 0.7)")
    print("  --log            Save execution logs to a json file")
    print("  --log-file path  Path to save the execution logs")


def print_synthesize_help() -> None:
    print_header("Synthesize")
    print("  Usage: uv run main.py synthesize {config} [config2 ...] [options]")
    print("         uv run main.py synthesize config list             List available configs")
    print("         uv run main.py synthesize config show {name}      Show config details")
    print("         uv run main.py synthesize config rm {name}        Delete a config")
    print("         uv run main.py synthesize key set {api_key}       Store OpenRouter API key")
    print("         uv run main.py synthesize key show                Show stored API key (masked)\n")
    print("  Generates synthetic Strata JSONL datasets using an AI model.")
    print("  Multiple configs can be passed as separate args or comma-separated.")
    print("  Output is saved to ~/.strata/datasets/{timestamp}_{id}.jsonl per config.\n")
    print("  Options")
    print("  --workers N        Concurrent API calls per config (default: 3)")
    print("  --multi-parallel   Run all configs simultaneously in parallel")
    print("  --resume path      Continue an existing .jsonl dataset (single config only)\n")
    print("  Standard config (~/.strata/configs/{name}.json)  mode: \"standard\"")
    print("  name          Config identifier")
    print("  description   Human-readable description")
    print("  mode          \"standard\"")
    print("  user_prompt   Instructions for the AI (what data to generate)")
    print("  model         OpenRouter model ID  (e.g. inclusionai/ling-2.6-flash)")
    print("  task          sft | chat | dpo | rl")
    print("  count         Total examples to generate")
    print("  temperature   Sampling temperature (0.0 – 2.0)")
    print("  batch_size    Examples requested per API call\n")
    print("  GFS config (~/.strata/configs/{name}.json)  mode: \"gfs\"")
    print("  name          Config identifier")
    print("  description   Human-readable description")
    print("  mode          \"gfs\"")
    print("  source        Path to a file or directory of source material")
    print("  glob          Optional glob filter for directories (e.g. \"**/*.py\")")
    print("  prompts       List of prompts — one pass per prompt per source unit")
    print("  model         OpenRouter model ID")
    print("  task          sft | chat | dpo | rl")
    print("  count         Examples per prompt per source unit")
    print("  temperature   Sampling temperature (0.0 – 2.0)")
    print("  batch_size    Examples requested per API call")
    print("  max_source_chars  Truncate source units to this length (default 8000)\n")
    print("  Dataset format (Strata JSONL — one JSON object per line)")
    print("  SFT:   {\"prompt\": \"...\", \"completion\": \"...\"}")
    print("  Chat:  {\"messages\": [{\"role\": \"user\", \"content\": \"...\"}, ...]}")
    print("  DPO:   {\"prompt\": \"...\", \"chosen\": \"...\", \"rejected\": \"...\"}")
    print("  RL:    {\"prompt\": \"...\"}")


import contextlib
import io

@contextlib.contextmanager
def suppress_stderr():
    """
    Redirects sys.stderr and low-level file descriptor 2 to suppress C-level logging
    from llama.cpp, PyTorch, CUDA, etc.
    """
    original_stderr = sys.stderr
    null_fd = None
    saved_stderr_fd = None
    try:
        null_fd = os.open(os.devnull, os.O_WRONLY)
    except Exception:
        pass

    try:
        stderr_fd = sys.stderr.fileno()
        saved_stderr_fd = os.dup(stderr_fd)
        if null_fd is not None:
            os.dup2(null_fd, stderr_fd)
    except Exception:
        pass

    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stderr = original_stderr
        if saved_stderr_fd is not None:
            try:
                os.dup2(saved_stderr_fd, stderr_fd)
                os.close(saved_stderr_fd)
            except Exception:
                pass
        if null_fd is not None:
            try:
                os.close(null_fd)
            except Exception:
                pass


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
                except ValueError as err:
                    raise ValueError(f"Option --max-tokens expects an integer, got '{args[i]}'") from err
        elif arg == "--temperature":
            i += 1
            if i < len(args):
                try:
                    opts["temperature"] = float(args[i])
                except ValueError as err:
                    raise ValueError(f"Option --temperature expects a float, got '{args[i]}'") from err
        else:
            positional.append(arg)
        i += 1

    if len(positional) >= 1:
        opts["model_name"] = positional[0]
    if len(positional) >= 2:
        opts["prompt"] = positional[1]

    return opts


def parse_train_args(args: list) -> dict:
    """
    Parse flags for the train command.
    Returns a dict matching TrainingConfig fields plus model_name.
    """
    opts = {
        "model_name": None,
        "data_path": None,
        "method": "lora",
        "run_name": None,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj", "v_proj"],
        "epochs": 3,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "warmup_steps": 10,
        "max_seq_length": 512,
        "weight_decay": 0.01,
        "seed": 42,
        "dpo_beta": 0.1,
        "grpo_beta": 0.04,
        "ppo_beta": 0.1,
        "reward_model": "",
        "reward_fn": "",
        "num_generations": 4,
        "max_new_tokens": 256,
        "use_lora": False,
        "resume": True,
    }

    positional = []
    i = 0
    while i < len(args):
        arg = args[i]

        def _next(flag: str) -> str:
            nonlocal i
            i += 1
            if i >= len(args):
                raise ValueError(f"Option {flag} expects a value.")
            return args[i]

        if arg == "--data":
            opts["data_path"] = _next("--data")
        elif arg == "--method":
            v = _next("--method")
            if v not in ("lora", "qlora", "full", "dpo", "grpo", "ppo"):
                raise ValueError(f"--method must be one of: lora, qlora, full, dpo, grpo, ppo. Got '{v}'.")
            opts["method"] = v
        elif arg == "--run-name":
            opts["run_name"] = _next("--run-name")
        elif arg == "--lora-r":
            opts["lora_r"] = int(_next("--lora-r"))
        elif arg == "--lora-alpha":
            opts["lora_alpha"] = int(_next("--lora-alpha"))
        elif arg == "--lora-dropout":
            opts["lora_dropout"] = float(_next("--lora-dropout"))
        elif arg == "--lora-modules":
            opts["lora_target_modules"] = [m.strip() for m in _next("--lora-modules").split(",")]
        elif arg == "--epochs":
            opts["epochs"] = int(_next("--epochs"))
        elif arg == "--batch-size":
            opts["batch_size"] = int(_next("--batch-size"))
        elif arg == "--grad-accum":
            opts["gradient_accumulation_steps"] = int(_next("--grad-accum"))
        elif arg == "--lr":
            opts["learning_rate"] = float(_next("--lr"))
        elif arg == "--warmup":
            opts["warmup_steps"] = int(_next("--warmup"))
        elif arg == "--max-seq":
            opts["max_seq_length"] = int(_next("--max-seq"))
        elif arg == "--weight-decay":
            opts["weight_decay"] = float(_next("--weight-decay"))
        elif arg == "--seed":
            opts["seed"] = int(_next("--seed"))
        elif arg == "--dpo-beta":
            opts["dpo_beta"] = float(_next("--dpo-beta"))
        elif arg == "--grpo-beta":
            opts["grpo_beta"] = float(_next("--grpo-beta"))
        elif arg == "--ppo-beta":
            opts["ppo_beta"] = float(_next("--ppo-beta"))
        elif arg == "--reward-model":
            opts["reward_model"] = _next("--reward-model")
        elif arg == "--reward-fn":
            opts["reward_fn"] = _next("--reward-fn")
        elif arg == "--num-generations":
            opts["num_generations"] = int(_next("--num-generations"))
        elif arg == "--max-new-tokens":
            opts["max_new_tokens"] = int(_next("--max-new-tokens"))
        elif arg == "--use-lora":
            opts["use_lora"] = True
        elif arg == "--no-resume":
            opts["resume"] = False
        else:
            positional.append(arg)
        i += 1

    if positional:
        opts["model_name"] = positional[0]

    return opts


def main() -> None:
    """
    command line entrypoint for strata.
    conforms to TreeSoft Terminal Interface (TTI) specification.
    """
    args = sys.argv[1:]

    if len(args) < 1:
        print_root_help()
        sys.exit(1)

    command: str = args[0].lower()

    # Handle help command globally
    if "--help" in args or "-h" in args:
        if command == "download":
            print_download_help()
        elif command == "list":
            print_list_help()
        elif command == "rm":
            print_rm_help()
        elif command == "rename":
            print_rename_help()
        elif command == "view":
            print_view_help()
        elif command == "run":
            print_run_help()
        elif command == "train":
            print_train_help()
        elif command == "synthesize":
            print_synthesize_help()
        else:
            print_root_help()
        sys.exit(0)

    valid_commands = {"download", "run", "list", "rm", "rename", "view", "train", "synthesize"}
    if command not in valid_commands:
        print_error("Help", f"Unknown command '{command}'.", "Run 'uv run main.py' to see the list of available commands.")
        sys.exit(1)

    if command == "list":
        if len(args) < 2:
            print_error("Models", "Subcommand for list is missing.", "Run 'uv run main.py list models' or 'uv run main.py list logs'.")
            sys.exit(1)
        subcommand = args[1].lower()
        if subcommand == "models":
            print_header("Models")
            manager = ModelManager()
            try:
                models = manager.get_available_models()
                if not models:
                    print("  No models downloaded yet.")
                else:
                    print("  Name".ljust(52) + "Type")
                    for m in models:
                        print(f"  {m['name'].lower().ljust(50)}  {m['type'].lower()}")
            except Exception as err:
                print_error("Models", str(err), "Check if the models directory exists or has correct permissions.")
                sys.exit(1)
        elif subcommand == "logs":
            print_header("Logs")
            try:
                logs = sorted(LOGS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                if not logs:
                    print("  No logs found.")
                else:
                    print("  Name".ljust(52) + "Size".ljust(12) + "Created")
                    for log in logs:
                        size = log.stat().st_size
                        if size < 1024:
                            size_str = f"{size} B"
                        else:
                            size_str = f"{size / 1024:.1f} KB"
                        mtime = log.stat().st_mtime
                        mtime_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))
                        print(f"  {log.name.ljust(50)}  {size_str.ljust(10)}  {mtime_str}")
            except Exception as err:
                print_error("Logs", str(err), "Check if the logs directory exists and is readable.")
                sys.exit(1)
        else:
            print_error("Models", f"Unknown list subcommand '{subcommand}'.", "Run 'uv run main.py list models' or 'uv run main.py list logs'.")
            sys.exit(1)
        sys.exit(0)

    elif command == "rm":
        if len(args) < 2:
            print_error("Models", "Model name is missing.", "Run 'uv run main.py rm {model_name}'.")
            sys.exit(1)
        model_name: str = args[1]
        print_header("Models")
        manager = ModelManager()
        try:
            manager.delete_model(model_name)
            print(f"  Model '{model_name.lower()}' removed.")
        except Exception as err:
            print(f"  ! Error: {str(err)}\n")
            print("  Run 'uv run main.py list models' to see available models.")
            sys.exit(1)
        sys.exit(0)

    elif command == "rename":
        if len(args) < 3:
            print_error("Models", "Old model name and new model name are required.", "Run 'uv run main.py rename {old_name} {new_name}'.")
            sys.exit(1)
        old_name: str = args[1]
        new_name: str = args[2]
        print_header("Models")
        manager = ModelManager()
        try:
            manager.rename_model(old_name, new_name)
            print(f"  Model '{old_name.lower()}' renamed to '{new_name.lower()}' successfully.")
        except Exception as err:
            print(f"  ! Error: {str(err)}\n")
            print("  Run 'uv run main.py list models' to check available models.")
            sys.exit(1)
        sys.exit(0)

    elif command == "download":
        if len(args) < 2:
            print_error("Download", "Download link is missing.", "Run 'uv run main.py download {link}'.")
            sys.exit(1)
        link: str = args[1]
        print_header("Download")
        try:
            from src.models.downloader import parse_hf_link
            parsed = parse_hf_link(link)
            repo_id = parsed["repo_id"]
            safe_repo_name = repo_id.replace("/", "--")
            dest_path = MODELS_DIR / safe_repo_name

            with suppress_stderr():
                download_model(
                    link=link,
                    dest_dir=MODELS_DIR,
                    progress_callback=lambda msg: print(f"  {msg}")
                )
            print(f"\n  Download completed.\n")
            print(f"  Model saved to {dest_path}")
        except Exception as err:
            print(f"\n  ! Error: {str(err)}\n")
            print("  Check your connection or Hugging Face link spelling.")
            sys.exit(1)
        sys.exit(0)

    elif command == "run":
        try:
            run_args = parse_run_args(args[1:])
        except ValueError as err:
            print_error("Run", str(err), "Check option syntax.")
            sys.exit(1)

        if run_args["model_name"] is None:
            print_error("Run", "Model name is missing.", "Run 'uv run main.py run {model_name} [prompt] [options]'.")
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
            print_error("Run", "Verbose and silent options are mutually exclusive.", "Select only one of --verbose or --silent.")
            sys.exit(1)

        print_header("Run")
        manager = ModelManager()
        try:
            ram_before = get_ram_usage_mb() if not bare else None
            sys_ram_before = get_system_ram_mb() if not bare else None

            # Loading progress
            print("  Loading model... ", end="", flush=True)
            with suppress_stderr():
                harness = manager.load_model(model_name)
            print("100%", flush=True)

            ram_after_load = get_ram_usage_mb() if not bare else None
            sys_ram_after_load = get_system_ram_mb() if not bare else None
            device = harness.device
            prompt_tokens = manager.count_tokens(prompt) if prompt else 0

            if verbose:
                ram_delta_str = f"{ram_after_load - ram_before:.1f} MB" if (ram_before is not None and ram_after_load is not None) else "n/a"
                sys_ram_str = f"{sys_ram_after_load['percent']:.1f}% used ({sys_ram_after_load['used']:.0f} MB / {sys_ram_after_load['total']:.0f} MB)" if sys_ram_after_load else "n/a"
                
                print()
                print("  Metadata\n")
                print(f"  Model:          {model_name.lower()}")
                print(f"  Device:         {device.lower()}")
                print(f"  Max tokens:     {max_tokens}")
                print(f"  Temperature:    {temperature}")
                print(f"  RAM load delta: {ram_delta_str}")
                print(f"  System RAM:     {sys_ram_str}")
                print(f"  Prompt tokens:  {prompt_tokens}")
                print()
            elif not silent:
                print()

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

            with suppress_stderr():
                stream = manager.generate_response_stream(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                for chunk in stream:
                    current_time = time.time()
                    dt = current_time - last_time
                    num_tokens = manager.count_tokens(chunk)
                    if num_tokens == 0 and chunk:
                        num_tokens = 1

                    total_tokens_so_far += num_tokens
                    elapsed = current_time - start_time

                    if silent:
                        print(f"\r  Generating... {total_tokens_so_far} tokens / {elapsed:.1f}s", end="", flush=True)
                    else:
                        print(chunk, end="", flush=True)

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
                print(f"\r  Generating... done ({total_tokens_so_far} tokens / {time.time() - start_time:.1f}s)", flush=True)
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
                print()
                print("  Metrics\n")
                
                speed_str = f"{tokens_per_second:.2f} tps ({peak_tps:.2f} peak / {lowest_tps:.2f} lowest)"
                
                if ram_after_load is not None and ram_after_inference is not None:
                    ram_str = f"{ram_after_inference:.1f} MB ({ram_after_inference - ram_after_load:+.1f} MB inference delta)"
                else:
                    ram_str = "n/a"
                    
                if sys_ram_final:
                    sys_ram_str = f"{sys_ram_final['percent']:.1f}% used ({sys_ram_final['used']:.0f} MB / {sys_ram_final['total']:.0f} MB)"
                else:
                    sys_ram_str = "n/a"

                print(f"  Time took:      {time_took:.2f}s")
                print(f"  Tokens:         {token_count}")
                print(f"  Speed:          {speed_str}")
                print(f"  Temperature:    {temp_str}")
                print(f"  RAM inference:  {ram_str}")
                print(f"  System RAM:     {sys_ram_str}")
                print()

                if verbose and chunk_token_log:
                    print("  Token Trace\n")
                    print("  Time".ljust(12) + "Tokens".ljust(10) + "Speed")
                    for entry in chunk_token_log:
                        elapsed_str = f"{entry['elapsed_s']:.3f}s"
                        speed_chunk_str = f"{entry['tps']:.2f} tps"
                        print(f"  {elapsed_str.ljust(10)}  {str(entry['tokens']).ljust(8)}  {speed_chunk_str}")
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
                    print(f"  Log saved to {log_file}\n")

        except Exception as err:
            print_error("Run", str(err), "Check model paths, hardware drivers, or options spelling.")
            sys.exit(1)
        sys.exit(0)

    elif command == "train":
        try:
            train_args = parse_train_args(args[1:])
        except ValueError as err:
            print_error("Train", str(err), "Run 'uv run main.py train --help' for usage.")
            sys.exit(1)

        if train_args["model_name"] is None:
            print_error("Train", "Model name is missing.", "Run 'uv run main.py train {model_name} --data {file.jsonl}'.")
            sys.exit(1)
        if not train_args["data_path"]:
            print_error("Train", "Dataset path is missing.", "Use --data {file.jsonl} to specify the training dataset.")
            sys.exit(1)

        print_header("Train")

        try:
            from src.training.config import TrainingConfig
            from src.training.trainer import run_training
            from src.training.dataset import load_dataset

            # resolve run name
            run_name = train_args["run_name"] or f"{train_args['model_name']}_{int(time.time())}"
            output_dir = str(TRAINED_DIR / run_name)

            # build config
            cfg = TrainingConfig(
                model_name=train_args["model_name"],
                run_name=run_name,
                data_path=train_args["data_path"],
                method=train_args["method"],
                lora_r=train_args["lora_r"],
                lora_alpha=train_args["lora_alpha"],
                lora_dropout=train_args["lora_dropout"],
                lora_target_modules=train_args["lora_target_modules"],
                epochs=train_args["epochs"],
                batch_size=train_args["batch_size"],
                gradient_accumulation_steps=train_args["gradient_accumulation_steps"],
                learning_rate=train_args["learning_rate"],
                warmup_steps=train_args["warmup_steps"],
                max_seq_length=train_args["max_seq_length"],
                weight_decay=train_args["weight_decay"],
                seed=train_args["seed"],
                dpo_beta=train_args["dpo_beta"],
                grpo_beta=train_args["grpo_beta"],
                ppo_beta=train_args["ppo_beta"],
                reward_model=train_args["reward_model"],
                reward_fn=train_args["reward_fn"],
                num_generations=train_args["num_generations"],
                max_new_tokens=train_args["max_new_tokens"],
                use_lora=train_args["use_lora"],
                resume=train_args["resume"],
                output_dir=output_dir,
            )

            # peek at dataset before training starts
            with suppress_stderr():
                ds_info = load_dataset(cfg.data_path)

            steps_per_epoch = max(
                1, (ds_info["count"] // (cfg.batch_size * cfg.gradient_accumulation_steps))
            )
            total_steps = steps_per_epoch * cfg.epochs

            print(f"  Model:      {cfg.model_name.lower()}")
            print(f"  Method:     {cfg.method}")
            print(f"  Task:       {ds_info['task']}")
            print(f"  Dataset:    {ds_info['count']} rows ({cfg.data_path})")
            print(f"  Epochs:     {cfg.epochs}")
            print(f"  Steps:      {total_steps}")
            print(f"  Run name:   {run_name}")
            print(f"  Output:     {output_dir}")
            if ds_info["skipped"]:
                print(f"\n  ~ Warning: {ds_info['skipped']} unrecognisable rows skipped.")
            print()

            # live progress: overwrite the current line each step
            _last_line_len = [0]

            def _progress(step: int, total: int, loss: float) -> None:
                line = f"  Step {step}/{total}   loss {loss:.4f}"
                pad = max(0, _last_line_len[0] - len(line))
                print(f"\r{line}{' ' * pad}", end="", flush=True)
                _last_line_len[0] = len(line)

            with suppress_stderr():
                final_dir = run_training(cfg, progress_callback=_progress)

            print(f"\r  {'Training complete.'.ljust(_last_line_len[0])}", flush=True)
            print()
            print(f"  Model saved to {final_dir}")

        except FileNotFoundError as err:
            print(f"\n  ! Error: {str(err)}\n")
            print("  Run 'uv run main.py list models' to see available models.")
            sys.exit(1)
        except Exception as err:
            print(f"\n  ! Error: {str(err)}\n")
            print("  Check your dataset format, model name, and available VRAM.")
            sys.exit(1)
        sys.exit(0)

    elif command == "synthesize":
        from src.synthesize.config import SynthConfig, GFSConfig, load_any_config, list_configs, delete_config
        import src.synthesize.openrouter as _or_provider
        import src.synthesize.agentrouter as _ar_provider
        from src.synthesize.generator import synthesize
        from src.synthesize.gfs import synthesize_gfs
        from src.config import DATASETS_DIR

        def _provider_for_key_cmd(provider_arg: str):
            """Return the provider module for key subcommands."""
            if provider_arg == "agentrouter":
                return _ar_provider, "AgentRouter"
            return _or_provider, "OpenRouter"

        sub_args = args[1:]

        # ---- key subcommand ------------------------------------------------
        # key set [provider] {api_key}   (provider defaults to openrouter)
        # key show [provider]
        if sub_args and sub_args[0].lower() == "key":
            print_header("Synthesize")
            if len(sub_args) < 2:
                print_error("Synthesize", "Key subcommand missing.", "Use 'key set [provider] {api_key}' or 'key show [provider]'.")
                sys.exit(1)
            key_sub = sub_args[1].lower()
            if key_sub == "set":
                # key set {api_key}  OR  key set {provider} {api_key}
                _known_providers = {"openrouter", "agentrouter"}
                if len(sub_args) >= 4 and sub_args[2].lower() in _known_providers:
                    _prov_mod, _prov_name = _provider_for_key_cmd(sub_args[2].lower())
                    _key_val = sub_args[3]
                elif len(sub_args) >= 3 and sub_args[2].lower() not in _known_providers:
                    _prov_mod, _prov_name = _provider_for_key_cmd("openrouter")
                    _key_val = sub_args[2]
                else:
                    print_error("Synthesize", "API key value is missing.", "Run 'uv run main.py synthesize key set [provider] {api_key}'.")
                    sys.exit(1)
                _prov_mod.save_key(_key_val)
                print(f"  {_prov_name} API key saved.")
            elif key_sub == "show":
                _prov_name_arg = sub_args[2].lower() if len(sub_args) >= 3 else "openrouter"
                _prov_mod, _prov_name = _provider_for_key_cmd(_prov_name_arg)
                if not _prov_mod.key_exists():
                    print(f"  No {_prov_name} API key stored.")
                else:
                    k = _prov_mod.load_key()
                    masked = k[:8] + "..." + k[-4:] if len(k) > 12 else "****"
                    print(f"  {_prov_name} API key: {masked}")
            else:
                print_error("Synthesize", f"Unknown key subcommand '{key_sub}'.", "Use 'key set [provider] {api_key}' or 'key show [provider]'.")
                sys.exit(1)
            sys.exit(0)

        # ---- config subcommand ---------------------------------------------
        if sub_args and sub_args[0].lower() == "config":
            print_header("Synthesize / Config")
            if len(sub_args) < 2:
                print_error("Synthesize", "Config subcommand missing.", "Use 'config list', 'config show {name}', or 'config rm {name}'.")
                sys.exit(1)
            cfg_sub = sub_args[1].lower()
            if cfg_sub == "list":
                entries = list_configs()
                if not entries:
                    print("  No configs found.")
                else:
                    print("  Name".ljust(32) + "Mode".ljust(12) + "Task".ljust(8) + "Count".ljust(8) + "Model")
                    for e in entries:
                        print(f"  {e['name'].ljust(30)}  {e['mode'].ljust(10)}  {str(e['task']).ljust(6)}  {str(e['count']).ljust(6)}  {e['model']}")
            elif cfg_sub == "show":
                if len(sub_args) < 3:
                    print_error("Synthesize", "Config name missing.", "Run 'uv run main.py synthesize config show {name}'.")
                    sys.exit(1)
                try:
                    c = load_any_config(sub_args[2])
                    print(f"  Name:        {c.name}")
                    print(f"  Description: {c.description}")
                    print(f"  Mode:        {c.mode}")
                    print(f"  Model:       {c.model}")
                    print(f"  Task:        {c.task}")
                    print(f"  Count:       {c.count}")
                    print(f"  Temperature: {c.temperature}")
                    print(f"  Batch size:  {c.batch_size}")
                    if isinstance(c, GFSConfig):
                        print(f"  Source:      {c.source}")
                        if c.glob:
                            print(f"  Glob:        {c.glob}")
                        print(f"  Max chars:   {c.max_source_chars}")
                        print(f"\n  Prompts ({len(c.prompts)}):")
                        for i, p in enumerate(c.prompts, 1):
                            print(f"    [{i}] {p}")
                    else:
                        print(f"\n  User prompt:")
                        for line in c.user_prompt.splitlines():
                            print(f"    {line}")
                except FileNotFoundError as err:
                    print_error("Synthesize", str(err), "Run 'uv run main.py synthesize config list' to see available configs.")
                    sys.exit(1)
            elif cfg_sub == "rm":
                if len(sub_args) < 3:
                    print_error("Synthesize", "Config name missing.", "Run 'uv run main.py synthesize config rm {name}'.")
                    sys.exit(1)
                try:
                    delete_config(sub_args[2])
                    print(f"  Config '{sub_args[2]}' deleted.")
                except FileNotFoundError as err:
                    print_error("Synthesize", str(err), "Run 'uv run main.py synthesize config list' to see available configs.")
                    sys.exit(1)
            else:
                print_error("Synthesize", f"Unknown config subcommand '{cfg_sub}'.", "Use 'config list', 'config show {name}', or 'config rm {name}'.")
                sys.exit(1)
            sys.exit(0)

        # ---- run synthesis -------------------------------------------------
        if not sub_args:
            print_synthesize_help()
            sys.exit(1)

        import random
        import string

        # Parse config names (positional, comma-separable) and flags
        config_names: list[str] = []
        synth_workers = 3
        multi_parallel = False
        resume_path: str | None = None
        _si = 0
        while _si < len(sub_args):
            _sa = sub_args[_si]
            if _sa == "--multi-parallel":
                multi_parallel = True
            elif _sa == "--workers":
                _si += 1
                if _si < len(sub_args):
                    try:
                        synth_workers = int(sub_args[_si])
                        if synth_workers < 1:
                            raise ValueError
                    except ValueError:
                        print_error("Synthesize", f"--workers expects a positive integer, got '{sub_args[_si]}'.", "Example: --workers 5")
                        sys.exit(1)
            elif _sa == "--resume":
                _si += 1
                if _si < len(sub_args):
                    resume_path = sub_args[_si]
                else:
                    print_error("Synthesize", "--resume expects a path to an existing .jsonl file.", "Example: --resume ~/.strata/datasets/20260101_120000_abc123.jsonl")
                    sys.exit(1)
            else:
                for _cn in _sa.split(","):
                    _cn = _cn.strip()
                    if _cn:
                        config_names.append(_cn)
            _si += 1

        if resume_path is not None and len(config_names) != 1:
            print_error("Synthesize", "--resume can only be used with a single config.", "Pass exactly one config name when resuming.")
            sys.exit(1)
        if resume_path is not None and multi_parallel:
            print_error("Synthesize", "--resume is incompatible with --multi-parallel.", "Resume only supports a single config.")
            sys.exit(1)

        if not config_names:
            print_synthesize_help()
            sys.exit(1)

        print_header("Synthesize")

        # Resolve resume_from before building the run list
        _resume_from = 0
        _resume_output: Path | None = None
        if resume_path is not None:
            _resume_output = Path(resume_path).expanduser()
            if not _resume_output.exists():
                print_error("Synthesize", f"Resume file not found: {_resume_output}", "Check the path and try again.")
                sys.exit(1)
            # Count valid lines already written
            try:
                _resume_from = sum(
                    1 for _line in _resume_output.read_text(encoding="utf-8").splitlines()
                    if _line.strip()
                )
            except Exception as err:
                print_error("Synthesize", f"Could not read resume file: {err}", "Check file permissions.")
                sys.exit(1)

        # Validate all configs and API keys upfront before starting anything
        _configs_to_run: list[tuple] = []
        for _cn in config_names:
            try:
                _cfg = load_any_config(_cn)
            except FileNotFoundError as err:
                print_error("Synthesize", str(err), "Run 'uv run main.py synthesize config list' to see available configs.")
                sys.exit(1)
            try:
                _cfg.validate()
            except ValueError as err:
                print_error("Synthesize", f"Config '{_cn}': {err}", "Edit the config file to fix the issue.")
                sys.exit(1)
            _is_ar = _cfg.model.startswith("agentrouter/")
            _prov = _ar_provider if _is_ar else _or_provider
            _prov_name = "AgentRouter" if _is_ar else "OpenRouter"
            if not _prov.key_exists():
                _key_cmd = "agentrouter" if _is_ar else "openrouter"
                print_error("Synthesize", f"Config '{_cn}' requires {_prov_name} but no API key is set.", f"Run 'uv run main.py synthesize key set {_key_cmd} {{api_key}}'.")
                sys.exit(1)
            if _resume_output is not None:
                _out = _resume_output
            else:
                _ts = time.strftime("%Y%m%d_%H%M%S")
                _rid = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
                _out = DATASETS_DIR / f"{_ts}_{_rid}.jsonl"
            _configs_to_run.append((_cfg, _out))

        import math

        # In parallel mode a lock serialises all stdout writes so lines
        # from different threads never interleave.
        _print_lock = threading.Lock()

        def _emit(text: str, lock: threading.Lock | None = None) -> None:
            if lock:
                with lock:
                    print(text, flush=True)
            else:
                print(text, flush=True)

        # Helper: run one config and stream per-batch logs to stdout.
        # log_batches   — print a line for every batch that completes.
        # lock          — if provided, all prints go through it (parallel mode).
        # Raises on error — caller handles it.
        def _run_one(cfg, output_path, log_batches: bool, lock: threading.Lock | None = None, resume_from: int = 0) -> dict:
            import math as _math
            _effective_target = max(0, cfg.count - resume_from)
            total_batches = _math.ceil(_effective_target / cfg.batch_size) if _effective_target > 0 else 0
            batch_num = [0]

            if isinstance(cfg, GFSConfig):
                if log_batches:
                    _emit("\n".join([
                        f"  Config:   {cfg.name}  —  {cfg.description}",
                        f"  Mode:     gfs",
                        f"  Model:    {cfg.model}",
                        f"  Task:     {cfg.task}",
                        f"  Source:   {cfg.source}",
                        f"  Prompts:  {len(cfg.prompts)}",
                        f"  Count:    {cfg.count} per prompt per source unit",
                        f"  Workers:  {synth_workers}",
                        f"  Output:   {output_path}",
                        "",
                    ]), lock)

                def _gfs_cb(src_idx, total_src, prm_idx, total_prm, generated, skipped):
                    batch_num[0] += 1
                    line = (
                        f"  {cfg.name}  batch {batch_num[0]}  "
                        f"source {src_idx}/{total_src}  "
                        f"prompt {prm_idx}/{total_prm}  "
                        f"{generated} generated"
                    )
                    if skipped:
                        line += f"  ({skipped} skipped)"
                    if log_batches:
                        _emit(line, lock)

                result = synthesize_gfs(cfg, output_path, progress_callback=_gfs_cb)
            else:
                if log_batches:
                    _resume_note = f"  Resuming: {resume_from} already generated\n" if resume_from > 0 else ""
                    _emit("\n".join([
                        f"  Config:   {cfg.name}  —  {cfg.description}",
                        f"  Mode:     standard",
                        f"  Model:    {cfg.model}",
                        f"  Task:     {cfg.task}",
                        f"  Target:   {cfg.count} examples",
                        f"  Workers:  {synth_workers}",
                        f"  Output:   {output_path}",
                        "",
                    ]) + _resume_note, lock)

                def _std_cb(generated: int, target: int, skipped: int) -> None:
                    batch_num[0] += 1
                    line = f"  {cfg.name}  batch {batch_num[0]}/{total_batches}  {generated}/{target} generated"
                    if skipped:
                        line += f"  ({skipped} skipped)"
                    if log_batches:
                        _emit(line, lock)

                result = synthesize(cfg, output_path, progress_callback=_std_cb, workers=synth_workers, resume_from=resume_from)

            # Dataset completion log — always printed (sequential or parallel)
            skipped_note = f"  ({result['skipped']} skipped)" if result["skipped"] else ""
            if isinstance(cfg, GFSConfig):
                src_note = f"  {result['sources']} sources"
            else:
                src_note = ""
            _emit(f"\n  * {cfg.name}  {result['generated']} examples{src_note}  {result['elapsed_s']:.1f}s{skipped_note}", lock)
            _emit(f"    Saved to {result['output']}", lock)

            return result

        if multi_parallel and len(_configs_to_run) > 1:
            # ---- multi-parallel: all configs run simultaneously -------------
            _emit(f"  Running {len(_configs_to_run)} configs in parallel  (workers per config: {synth_workers})\n")
            for _cfg, _ in _configs_to_run:
                _emit(f"  - {_cfg.name}")
            _emit("")

            _thread_results: list = [None] * len(_configs_to_run)
            _thread_errors: list = [None] * len(_configs_to_run)

            def _thread_target(idx: int, cfg, out) -> None:
                try:
                    _thread_results[idx] = _run_one(cfg, out, log_batches=True, lock=_print_lock)
                except Exception as exc:
                    _thread_errors[idx] = exc
                    with _print_lock:
                        print(f"\n  ! {cfg.name}: {exc}", flush=True)

            _threads = [
                threading.Thread(target=_thread_target, args=(i, cfg, out), daemon=False)
                for i, (cfg, out) in enumerate(_configs_to_run)
            ]
            for _t in _threads:
                _t.start()
            for _t in _threads:
                _t.join()

            _emit("")
            if any(e is not None for e in _thread_errors):
                sys.exit(1)

        else:
            # ---- sequential: one config at a time ---------------------------
            for _idx, (_cfg, _out) in enumerate(_configs_to_run):
                if len(_configs_to_run) > 1:
                    _emit(f"  [{_idx + 1}/{len(_configs_to_run)}] {_cfg.name}\n")

                try:
                    _run_one(_cfg, _out, log_batches=True, resume_from=_resume_from)
                except (FileNotFoundError, ValueError) as err:
                    _emit(f"\n  ! Error: {err}\n")
                    _emit("  Check the 'source' path and glob pattern in your config.")
                    sys.exit(1)
                except RuntimeError as err:
                    _emit(f"\n  ! Error: {err}\n")
                    _emit("  Check your API key and model name.")
                    sys.exit(1)
                except Exception as err:
                    _emit(f"\n  ! Error: {err}\n")
                    sys.exit(1)

                if len(_configs_to_run) > 1 and _idx < len(_configs_to_run) - 1:
                    _emit("")

        sys.exit(0)

    elif command == "view":
        if len(args) < 2:
            print_error("Logs", "Log filename or path is missing.", "Run 'uv run main.py view {filename | path}'.")
            sys.exit(1)
        target = args[1]
        target_path = Path(target)
        if not target_path.exists():
            candidate = LOGS_DIR / target
            if candidate.exists():
                target_path = candidate
            else:
                print_error("Logs", f"File not found: {target}", "Run 'uv run main.py list logs' to check available logs.")
                sys.exit(1)

        print_header("Logs")
        print(f"  Opening log file {target_path.name}...")

        import subprocess
        if sys.platform == "win32":
            try:
                subprocess.run(["notepad.exe", str(target_path)])
            except Exception as err:
                print(f"\n  ! Error: Could not open notepad: {str(err)}\n")
                print("  Try opening the log file path manually.")
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
                print(f"\n  ! Error: No suitable editor found (nano, vim, vi)\n")
                print("  Try opening the log file path manually.")
                sys.exit(1)
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
