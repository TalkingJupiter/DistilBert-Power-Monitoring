import matplotlib.pyplot as plt
import pandas as pd
import json

json_file = "power_log_20250730_163941.jsonl"

data = []
with open(json_file, "r") as f:
    for line in f:
        data.append(json.loads(line.strip()))

df = pd.DataFrame(data)
df["timestamp"] = pd.to_datetime(df["timestamp"])

plt.figure(figsize=(14, 6))
plt.plot(df["timestamp"], df["power_watts"], label="GPU Power (W)")
plt.plot(df["timestamp"], df["cpu_utilization_percent"], label="CPU Utilization (%)")
plt.plot(df["timestamp"], df["gpu_utilization_percent"], label="GPU Utilization (%)")
plt.xlabel("Timestamp")
plt.ylabel("Metrics Values")
plt.title("System Metrics Over Time")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.savefig("system_metrics.png")
# plt.show()


