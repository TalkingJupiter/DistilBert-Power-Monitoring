import time
import json
import argparse
import signal
from datetime import datetime
from pynvml import *
import psutil

# Global flag for clean shutdown
running = True

def handle_signal(signum, frame):
    global running
    print(f"\n[INFO] Received signal {signum}. Stopping monitoring...")
    running = False

def monitor_power(gpu_index=0, interval=5, log_path="logs/power_log.jsonl"):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(gpu_index)

    print(f"[INFO] Starting GPU/CPU monitoring on GPU {gpu_index} every {interval}s.")
    print(f"[INFO] Logging to: {log_path}")

    with open(log_path, "a") as f:
        while running:
            try:
                # GPU metrics
                power = nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                mem = nvmlDeviceGetMemoryInfo(handle)
                util = nvmlDeviceGetUtilizationRates(handle)

                # CPU utilization
                cpu_util = psutil.cpu_percent(interval=None)

                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "gpu_index": gpu_index,
                    "power_watts": power,
                    "memory_used_MB": mem.used / 1024**2,
                    "gpu_utilization_percent": util.gpu,
                    "memory_utilization_percent": util.memory,
                    "cpu_utilization_percent": cpu_util
                }

                f.write(json.dumps(entry) + "\n")
                f.flush()
                time.sleep(interval)

            except Exception as e:
                print(f"[ERROR] {e}")
                break

    nvmlShutdown()
    print("[INFO] Monitoring stopped and NVML shutdown completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Power, Utilization, and CPU Monitor")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to monitor")
    parser.add_argument("--interval", type=int, default=5, help="Polling interval in seconds")
    parser.add_argument("--log_path", type=str, default="logs/power_log.jsonl", help="Output log file")

    args = parser.parse_args()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    monitor_power(gpu_index=args.gpu, interval=args.interval, log_path=args.log_path)
