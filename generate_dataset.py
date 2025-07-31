from datasets import load_dataset
import os

print("[INFO] Loading OSCAR English dataset (10%)...")
dataset = load_dataset("oscar", "unshuffled_deduplicated_en", split="train[:10%]")

output_dir = "data"
output_file = os.path.join(output_dir, "dump.txt")

print(f"[INFO] Saving extracted text to {output_file}...")
os.makedirs(output_dir, exist_ok=True)

with open(output_file, "w", encoding="utf-8") as f:
    for i, item in enumerate(dataset):
        text = item.get("text", "").strip()
        if text:
            f.write(text.replace("\n", " ") + "\n")
        if (i + 1) % 10000 == 0:
            print(f"[INFO] Written {i+1} examples")

print("[INFO] Dataset extraction complete.")
