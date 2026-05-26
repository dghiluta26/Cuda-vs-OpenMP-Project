from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
results_dir = project_root / "results"
graphs_dir = project_root / "report" / "graphs"
graphs_dir.mkdir(parents=True, exist_ok=True)

thread_files = {
    1: results_dir / "results_release_threads_1.csv",
    2: results_dir / "results_release_threads_2.csv",
    4: results_dir / "results_release_threads_4.csv",
    8: results_dir / "results_release_threads_8.csv",
    16: results_dir / "results_release_threads_16.csv",
}

missing_files = [path for path in thread_files.values() if not path.exists()]
if missing_files:
    missing = "\n".join(str(path) for path in missing_files)
    raise FileNotFoundError(f"Missing benchmark CSV files:\n{missing}")

all_data = []

for threads, file_path in thread_files.items():
    df = pd.read_csv(file_path)
    df["test_threads"] = threads
    all_data.append(df)

df_all = pd.concat(all_data, ignore_index=True)
df_all = df_all.sort_values(["total_pixels", "method", "test_threads"])


def add_resolution_labels(df):
    labels = df[["total_pixels", "width", "height"]].drop_duplicates()
    labels = labels.sort_values("total_pixels")
    ticks = labels["total_pixels"].tolist()
    names = [f"{row.width}x{row.height}" for row in labels.itertuples()]
    plt.xticks(ticks, names, rotation=20)


def plot_stage_time(column, ylabel, title, output_name):
    plt.figure(figsize=(8, 5))
    for method in ["sequential", "openmp", "cuda"]:
        subset = df_main[df_main["method"] == method].sort_values("total_pixels")
        plt.plot(subset["total_pixels"], subset[column], marker="o", label=method)

    add_resolution_labels(df_main)
    plt.xlabel("Image resolution")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(graphs_dir / output_name, dpi=150)
    plt.close()

# Use the 16-thread run as the main comparison between sequential, OpenMP and CUDA
df_main = df_all[df_all["test_threads"] == 16]

# ---------------------------------------------------------
# Graph 1: Total time vs resolution
# ---------------------------------------------------------
plot_stage_time(
    "total_ms",
    "Total time (ms)",
    "Total execution time vs image resolution",
    "total_time_vs_resolution.png",
)

# ---------------------------------------------------------
# Graph 2: Histogram time vs resolution
# ---------------------------------------------------------
plot_stage_time(
    "histogram_ms",
    "Histogram time (ms)",
    "Histogram computation time vs image resolution",
    "histogram_time_vs_resolution.png",
)

# ---------------------------------------------------------
# Graph 3: Thresholding time vs resolution
# ---------------------------------------------------------
plot_stage_time(
    "thresholding_ms",
    "Thresholding time (ms)",
    "Thresholding time vs image resolution",
    "thresholding_time_vs_resolution.png",
)

# ---------------------------------------------------------
# Graph 4: OpenMP speedup vs number of threads
# Use the largest image because scaling is clearer there.
# ---------------------------------------------------------
largest_pixels = df_all["total_pixels"].max()
df_largest = df_all[
    (df_all["total_pixels"] == largest_pixels) &
    (df_all["method"] == "openmp")
]
df_largest = df_largest.sort_values("test_threads")

plt.figure(figsize=(8, 5))
plt.plot(df_largest["test_threads"], df_largest["speedup"], marker="o")
plt.xlabel("OpenMP threads")
plt.ylabel("Speedup vs sequential")
plt.title("OpenMP speedup vs number of threads")
plt.xticks(sorted(thread_files.keys()))
plt.grid(True, alpha=0.35)
plt.tight_layout()
plt.savefig(graphs_dir / "openmp_speedup_vs_threads.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# Graph 5: CUDA speedup vs resolution
# ---------------------------------------------------------
df_cuda = df_main[df_main["method"] == "cuda"].sort_values("total_pixels")

plt.figure(figsize=(8, 5))
plt.plot(df_cuda["total_pixels"], df_cuda["speedup"], marker="o")
add_resolution_labels(df_cuda)
plt.xlabel("Image resolution")
plt.ylabel("CUDA speedup vs sequential")
plt.title("CUDA speedup vs image resolution")
plt.grid(True, alpha=0.35)
plt.tight_layout()
plt.savefig(graphs_dir / "cuda_speedup_vs_resolution.png", dpi=150)
plt.close()

print(f"Graphs saved to: {graphs_dir}")
