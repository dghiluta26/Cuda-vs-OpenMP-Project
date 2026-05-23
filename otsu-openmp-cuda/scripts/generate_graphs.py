from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

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

all_data = []

for threads, file_path in thread_files.items():
    df = pd.read_csv(file_path)
    df["test_threads"] = threads
    all_data.append(df)

df_all = pd.concat(all_data, ignore_index=True)

# Use the 16-thread run as the main comparison between sequential, OpenMP and CUDA
df_main = df_all[df_all["test_threads"] == 16]

# ---------------------------------------------------------
# Graph 1: Total time vs resolution
# ---------------------------------------------------------
plt.figure()
for method in ["sequential", "openmp", "cuda"]:
    subset = df_main[df_main["method"] == method]
    plt.plot(subset["total_pixels"], subset["total_ms"], marker="o", label=method)

plt.xlabel("Total pixels")
plt.ylabel("Total time (ms)")
plt.title("Total execution time vs image resolution")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(graphs_dir / "total_time_vs_resolution.png")
plt.close()

# ---------------------------------------------------------
# Graph 2: Histogram time vs resolution
# ---------------------------------------------------------
plt.figure()
for method in ["sequential", "openmp", "cuda"]:
    subset = df_main[df_main["method"] == method]
    plt.plot(subset["total_pixels"], subset["histogram_ms"], marker="o", label=method)

plt.xlabel("Total pixels")
plt.ylabel("Histogram time (ms)")
plt.title("Histogram computation time vs image resolution")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(graphs_dir / "histogram_time_vs_resolution.png")
plt.close()

# ---------------------------------------------------------
# Graph 3: Thresholding time vs resolution
# ---------------------------------------------------------
plt.figure()
for method in ["sequential", "openmp", "cuda"]:
    subset = df_main[df_main["method"] == method]
    plt.plot(subset["total_pixels"], subset["thresholding_ms"], marker="o", label=method)

plt.xlabel("Total pixels")
plt.ylabel("Thresholding time (ms)")
plt.title("Thresholding time vs image resolution")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(graphs_dir / "thresholding_time_vs_resolution.png")
plt.close()

# ---------------------------------------------------------
# Graph 4: OpenMP speedup vs number of threads
# Use the largest image because scaling is clearer there.
# ---------------------------------------------------------
largest_pixels = df_all["total_pixels"].max()
df_largest = df_all[
    (df_all["total_pixels"] == largest_pixels) &
    (df_all["method"] == "openmp")
]

plt.figure()
plt.plot(df_largest["test_threads"], df_largest["speedup"], marker="o")
plt.xlabel("OpenMP threads")
plt.ylabel("Speedup vs sequential")
plt.title("OpenMP speedup vs number of threads")
plt.grid(True)
plt.tight_layout()
plt.savefig(graphs_dir / "openmp_speedup_vs_threads.png")
plt.close()

# ---------------------------------------------------------
# Graph 5: CUDA speedup vs resolution
# ---------------------------------------------------------
df_cuda = df_main[df_main["method"] == "cuda"]

plt.figure()
plt.plot(df_cuda["total_pixels"], df_cuda["speedup"], marker="o")
plt.xlabel("Total pixels")
plt.ylabel("CUDA speedup vs sequential")
plt.title("CUDA speedup vs image resolution")
plt.grid(True)
plt.tight_layout()
plt.savefig(graphs_dir / "cuda_speedup_vs_resolution.png")
plt.close()

print(f"Graphs saved to: {graphs_dir}")