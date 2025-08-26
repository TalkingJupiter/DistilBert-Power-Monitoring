# DistilBert Power Monitoring
Author: @TalkingJupiter, @VictorSanh

This repo has been built original repo of [Distil*](https://github.com/huggingface/transformers-research-projects/tree/main/distillation). The purpose of this repo is to keep track the energy usage of the distillation process.

## Changes on the original Distil* Repo
We faced with a few problems on the Distil* repo while we are trying to run the scripts without any modification or change. One of the problems we have faced is the numpy libary's matrix part refused to work on the orginal repo. I forked the repo and moded the original code to work.

## How do we monitor the Distillation Process
The GPU/CPU (and optional per-process) monitor initializes a global `running` flag set to true and registers a `handle_signal` function that logs a stop message and flips `running` to false on SIGINT/SIGTERM. On start, `monitor_power(gpu_index, interval_seconds, log_path, pid)` initializes NVML, acquires a handle for the selected GPU, announces that monitoring has begun, and opens the log file for appending. If a PID is provided, it attempts to create a `psutil.Process` for that PID and primes per-process CPU measurement; if the PID is missing it logs an error and disables per-process metrics. It then enters a loop that continues while `running` is true: each tick it stamps the current time, reads device-level GPU data (power, memory info, and utilization) via NVML, and reads system-wide CPU utilization via psutil. If per-process tracking is active, it also captures the process’s CPU% and RSS memory, and queries NVML’s “compute running processes” to find the matching PID and record its GPU memory usage when available; if the process has exited, it logs that fact and stops per-process sampling. All collected fields are assembled into a single JSON object—always including timestamp, gpu\_index, power\_watts, memory\_used\_MB, gpu\_utilization\_percent, memory\_utilization\_percent, and cpu\_utilization\_percent, plus a nested `process` block when a PID is tracked—then written as one JSONL line, flushed, and the loop sleeps for the configured interval. Any exception during sampling is logged as an error and breaks the loop. On exit, the log file is closed, NVML is shut down, and a completion message is printed. The program accepts `--gpu` (default 0), `--interval` (seconds, default 5), `--log_path` (default `logs/power_log.jsonl`), and optional `--pid`, registers the signal handlers, and invokes `monitor_power` with the parsed arguments.

### Pseudocode for monitoring process
```bash
TITLE: GPU/CPU (and optional per-process) monitor — pseudocode

GLOBAL:
  running ← true

FUNCTION handle_signal(signal):
  PRINT "[INFO] Received signal, stopping monitoring…"
  running ← false

FUNCTION monitor_power(gpu_index, interval_seconds, log_path, pid = null):
  INIT NVML
  gpu_handle ← NVML.get_handle_by_index(gpu_index)

  PRINT "[INFO] Starting monitoring…"
  OPEN log_path FOR append AS logfile

  # Optional: set up per-process tracking
  proc ← null
  IF pid IS NOT null:
    TRY:
      proc ← psutil.Process(pid)
      proc.cpu_percent(None)    # prime per-process CPU measurement
    CATCH NoSuchProcess:
      PRINT "[ERROR] PID not found; continuing without per-process metrics"
      pid ← null

  # Main sampling loop
  WHILE running IS true:
    TRY:
      now_iso ← CURRENT_TIME_ISO8601()

      # -------- Device-level GPU stats (NVML) --------
      power_watts ← NVML.power_usage(gpu_handle) / 1000
      mem_info ← NVML.memory_info(gpu_handle)         # {used, total}
      util ← NVML.utilization_rates(gpu_handle)       # {gpu%, memory%}

      # -------- System CPU --------
      cpu_util_pct ← psutil.cpu_percent(None)

      # -------- Optional: per-process stats --------
      proc_cpu_pct ← null
      proc_mem_rss_mb ← null
      proc_gpu_mem_mb ← null

      IF proc IS NOT null:
        TRY:
          proc_cpu_pct ← proc.cpu_percent(None)                    # %
          proc_mem_rss_mb ← proc.memory_info().rss / 1024^2       # MB

          # Per-process GPU memory via NVML
          compute_procs ← NVML.get_compute_running_processes(gpu_handle)  # list of {pid, usedGpuMemory}
          MATCH item IN compute_procs WHERE item.pid == pid:
            IF item.usedGpuMemory IS available:
              proc_gpu_mem_mb ← item.usedGpuMemory / 1024^2

        CATCH NoSuchProcess:
          PRINT "[INFO] Target process exited"
          proc ← null

      # -------- Build one JSONL record --------
      entry ← {
        "timestamp": now_iso,
        "gpu_index": gpu_index,
        "power_watts": power_watts,
        "memory_used_MB": mem_info.used / 1024^2,
        "gpu_utilization_percent": util.gpu,
        "memory_utilization_percent": util.memory,
        "cpu_utilization_percent": cpu_util_pct
      }

      IF pid IS NOT null:
        entry["process"] ← {
          "pid": pid,
          "cpu_utilization_percent": proc_cpu_pct,
          "memory_rss_MB": proc_mem_rss_mb,
          "gpu_memory_used_MB": proc_gpu_mem_mb
        }

      WRITE JSON(entry) + "\n" TO logfile
      FLUSH logfile
      SLEEP interval_seconds

    CATCH Exception e:
      PRINT "[ERROR] " + e.message
      BREAK

  CLOSE logfile
  NVML.SHUTDOWN()
  PRINT "[INFO] Monitoring stopped"

# --- Program entrypoint ---
PARSE ARGS:
  --gpu (int, default 0)
  --interval (int seconds, default 5)
  --log_path (string, default "logs/power_log.jsonl")
  --pid (int, optional)

REGISTER signal handlers:
  SIGINT → handle_signal
  SIGTERM → handle_signal

CALL monitor_power(gpu_index, interval, log_path, pid)

```


 <br>

# How to Run?
## Option 1: Fully Automatic SH file
To run the experiment on your device just run `submit_both.sh` file. There might be some errors due to system related if these errors occurs please use other options.

## Option 2: Via submitting SH files
Since this project build in the [REPACSS](https://repacss.org/) Data Center we used `sbatch` command to submit our jobs, but depending on your system you can customize the files. 

1. Submit the `preprocess.sh` file. This will create the dataset to start the distillation process. 
2. After `preprocess.sh` job completes, submit `distill_experiment.sh` file
3. After the job completes it is ready to process the data.

## Option 3: Run Manually
> ⚠️⚠️ IMPORTANT: It may take more than 12 hours to run the whole experiment. We do not advise you to run the experiment manually.



1. Generate the dump.txt file
    ```bash
    python generate_dataset.py
    ```
2. Binarize the `dump.txt` file
    ```bash
      python scripts/binarized_data.py \
        --file_path data/dump.txt \
        --tokenizer_type bert \
        --tokenizer_name bert-base-uncased \
        --dump_file data/binarized_text
    ```
3. Tokenize
    ```bash
      python scripts/token_counts.py \
        --data_file data/binarized_text.bert-base-uncased.pickle \
        --token_counts_dump data/token_counts.bert-base-uncased.pickle \
        --vocab_size 30522
    ```
4. Run the Distillation
    > Note: Don't forget to change the `<EXPERIMENT NUMBER>` in the command arguments


    ```bash
    python train.py \
      --student_type distilbert \
      --student_config training_configs/distilbert-base-uncased.json \
      --teacher_type bert \
      --teacher_name bert-base-uncased \
      --mlm \
      --alpha_ce 0.5 \
      --alpha_mlm 0.5 \
      --alpha_clm 0.0 \
      --mlm_mask_prop 0.15 \
      --batch_size 5 \
      --n_epoch 3 \
      --data_file data/binarized_text.bert-base-uncased.pickle \
      --token_counts data/token_counts.bert-base-uncased.pickle \
      --dump_path serialization_dir/exp_<EXPERIMENT NUMBER> \
      --force
    ```