import numpy as np
import os
import matplotlib.pyplot as plt
from collections import Counter



def pad_six_digits(n: int) -> str:
    return f"{n:06d}"
def highlight_top_percent(data, percent):
    magnitudes = np.abs(data)

    # Calculate the threshold for the top x%
    threshold = np.percentile(magnitudes, 100 - percent)

    # Find indices where magnitude exceeds threshold
    rows, cols = np.where(magnitudes >= threshold)

    # Create the plot
    plt.figure(figsize=(20, 20 * data.shape[0] / data.shape[1]))  # Adjust aspect ratio
    plt.scatter(cols, rows, c='black', s=1)  # s controls dot size

    # Set the plot limits to match matrix dimensions
    plt.xlim(-0.5, data.shape[1] - 0.5)
    plt.ylim(-0.5, data.shape[0] - 0.5)

    # Invert y-axis so (0,0) is at top-left (matrix convention)
    plt.gca().invert_yaxis()

    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.title(f'Top {percent}% of values by magnitude')
    plt.grid(True, alpha=0.3)


def top_l2_indices(a: np.ndarray, percent=1, dim: str = "column") -> np.ndarray:
    A = np.asarray(a)
    if A.ndim != 2:
        raise ValueError("Input must be a 2D array")

    dim_norm = dim.lower()
    if dim_norm in ("column", "col", "c"):
        axis = 0  # compute norms per column
    elif dim_norm in ("row", "r"):
        axis = 1  # compute norms per row
    else:
        raise ValueError("dim must be 'column'/'col'/'c' or 'row'/'r'")

    size = A.shape[1 - axis]  # Fixed: number of columns or rows we're selecting from
    k = max(1, min(size, int(np.ceil(float(percent)/100 * size))))

    # Use squared norms (monotonic with L2, avoids sqrt)
    norms_sq = np.sum(A**2, axis=axis)

    # Indices of k largest norms (unsorted)
    topk_unsorted = np.argpartition(norms_sq, -k)[-k:]
    # Sort them by decreasing norm
    order = np.argsort(norms_sq[topk_unsorted])[::-1]
    return topk_unsorted[order]

def plot_dict_values(data):
    """
    Plots a scatter plot where x-axis is dictionary keys and 
    y-axis shows all values from the corresponding arrays.
    
    Parameters:
    data (dict): Dictionary with numeric keys and list values
    """
    x_values = []
    y_values = []
    
    # Flatten the data structure
    for key, values in data.items():
        for value in values:
            x_values.append(key)
            y_values.append(value)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x_values, y_values, c='black', s=1)
    plt.xlabel('Keys')
    plt.ylabel('Values')
    plt.title('Dictionary Values Plot')
    plt.grid(True, alpha=0.3)

def count_values(data):
    """
    Counts unique numbers and their frequencies across all arrays in the dictionary.
    
    Parameters:
    data (dict): Dictionary with numeric keys and list values
    
    Returns:
    tuple: (number of unique values, list of (value, count) tuples sorted by count descending)
    """
    # Flatten all values into a single list
    all_values = []
    for values in data.values():
        all_values.extend(values)
    
    # Count occurrences of each value
    value_counts = Counter(all_values)
    
    # Sort by count (largest to smallest)
    sorted_counts = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Number of unique values
    num_unique = len(value_counts)
    
    return num_unique, sorted_counts



DIR = "/share/desa/nfs02/cold/jamal-runs-benckmarking/Qwen_Qwen2.5-0.5B-mnli/0.0/grad_dump"
HIGHLIGHT_PERCENT=0.5

indices_by_step = {}
axis = "column"
LAYER="L11_mlp_up_proj_weight.npy"
directory = f"jamalplots/{LAYER.replace('.npy', '')}"
os.makedirs(directory, exist_ok=True)
for i in range(0, 500, 100):
    STEP = pad_six_digits(i)
    data = np.load(os.path.join(DIR, f"step{STEP}", LAYER))
    if i % 100 == 0:
        highlight_top_percent(data, HIGHLIGHT_PERCENT)
        plt.savefig(os.path.join(directory, f"step_{STEP}_top{HIGHLIGHT_PERCENT}_{axis}.png"), dpi=300)
    topnorms = top_l2_indices(data, percent=HIGHLIGHT_PERCENT, dim=axis)
    indices_by_step[STEP] = topnorms

    #print(f"Step {STEP}, Top {HIGHLIGHT_PERCENT}% columns indices: {topcols}")

plot_dict_values(indices_by_step)
plt.savefig(os.path.join(directory, f"top{HIGHLIGHT_PERCENT}_{axis}.png"), dpi=300)

num_unique, counts = count_values(indices_by_step)
print(f"Number of unique values: {num_unique}")
print("\nValue counts (sorted by frequency):")
for value, count in counts:
    print(f"  {value}: appears {count} times")