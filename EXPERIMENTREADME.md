# GPU and CPU Monitoring Tool

This Python-based tool monitors **GPU power consumption**, **GPU utilization**, **GPU memory usage**, and **CPU utilization** in real-time. It logs all metrics to a JSON Lines (`.jsonl`) file for later analysis.

---

## 📌 Features

- 🧠 **GPU Power Monitoring**: Uses NVIDIA Management Library (NVML) to monitor real-time power draw in watts.
- 💾 **GPU Memory Usage**: Logs memory used (in MB) on the selected GPU.
- 📊 **GPU Utilization**: Logs percentage of GPU compute and memory usage.
- 🖥️ **CPU Utilization**: Logs system-wide CPU usage percentage.
- 🧾 **Structured Output**: Logs each data sample as a JSON object to a `.jsonl` file.
- 🛑 **Graceful Exit**: Cleanly shuts down and finalizes NVML upon receiving SIGINT or SIGTERM.
- 🛠️ **Fully Configurable**: Set polling interval, GPU index, and log file path via command-line arguments.

---

## 📦 Requirements

- Python 3.7+
- [pynvml](https://pypi.org/project/nvidia-ml-py3/)
- [psutil](https://pypi.org/project/psutil/)

### Install dependencies

```bash
pip install pynvml psutil
```

---

## 🚀 Usage

```bash
python monitor.py --gpu 0 --interval 5 --log_path logs/power_log.jsonl
```

### Available Arguments

| Argument       | Description                                     | Default                     |
|----------------|-------------------------------------------------|-----------------------------|
| `--gpu`        | GPU index to monitor (starting from 0)          | `0`                         |
| `--interval`   | Time in seconds between samples                  | `5`                         |
| `--log_path`   | File path to save logs in JSON Lines format      | `logs/power_log.jsonl`      |

---

## 🧪 Sample Output

Each entry in the log file looks like:

```json
{
  "timestamp": "2025-07-30T17:01:23.456789",
  "gpu_index": 0,
  "power_watts": 135.7,
  "memory_used_MB": 8192.0,
  "gpu_utilization_percent": 76,
  "memory_utilization_percent": 48,
  "cpu_utilization_percent": 22.5
}
```

This format makes it easy to analyze with tools like:
- `pandas` for Python
- `jq` on the command line
- Data visualization tools like Grafana or Kibana (with preprocessing)

---

## 📂 Recommended Directory Structure

```
gpu-cpu-monitor/
├── monitor.py
├── logs/
│   └── power_log.jsonl
├── README.md
```

---

## 🧹 Clean Exit & Signal Handling

The tool supports graceful shutdown on:
- `Ctrl+C` (SIGINT)
- `kill` command (SIGTERM)

This ensures the log file is properly closed and the NVML interface is cleanly shutdown.

---

## 🔍 Debugging Tips

- If you get an error like `NVMLError_LibraryNotFound`, ensure:
  - You have an NVIDIA GPU and the driver is installed
  - You're running on a system with access to NVIDIA libraries

- If you're not seeing memory utilization:
  - Confirm the GPU is active with a running workload

---

## 🛠️ Future Enhancements (PRs Welcome!)

- [ ] Multi-GPU support (log all GPUs in a loop)
- [ ] Log CPU utilization per core
- [ ] CSV and SQLite export options
- [ ] Real-time terminal dashboard (e.g., curses-based UI)
- [ ] REST API endpoint to expose metrics

---

## 👨‍💻 Author

Developed by [Your Name], HPC researcher at [Your Institution].

---

## 📜 License

This project is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).
