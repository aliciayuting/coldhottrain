from collections import Counter
import os
from pprint import pprint
import torch
import json

# #with open("/share/desa/nfs02/cold/test/preselect_grads_2p.json", "r") as f:
# with open("./masks/preselect_grads_adam_0.08.json", "r") as f:
#     preselect_data = json.load(f)
# print(preselect_data['layers']['L00_self_attn_v_proj_weight.npy'])
# print(preselect_data['percent'])
# exit(0)

PATH = "/home/jah649/other-coldhottrain/analyze/adam_norms/stats"

top_percent = 0.04
equal_between_layers = True
train_all_bias = False
shape = [768, 768]
aggregate_stats = {}
mask_name = f"preselect_grads_adam_{top_percent}_{'equal' if equal_between_layers else 'unequal'}_train_bias_{train_all_bias}.json"
def roberta_layer_to_preselect_name(layer_name: str) -> str:
    # Example: "roberta.encoder.layer.0.attention.self.query.weight" -> "L00_self_attn_q_proj_weight.npy"
    parts = layer_name.split('.')
    if len(parts) != 8:
        raise ValueError(f"Unexpected layer name format: {layer_name}")
    layer_num = int(parts[3])
    q_or_v = parts[6]
    if q_or_v == 'query':
        qv_str = 'q'
    elif q_or_v == 'value':
        qv_str = 'v'
    else:
        raise ValueError(f"Unexpected q/v part: {q_or_v}")
    preselect_name = f"L{layer_num:02d}_self_attn_{qv_str}_proj_weight.npy"
    return preselect_name

#code to process every file in a folder
for filename in os.listdir(PATH):
    if not filename.endswith(".pt"):  # process only .pt files
        continue
    file_path = os.path.join(PATH, filename)
    data = torch.load(file_path)
    layer = data['param_name']
    if 'classifier' in layer or 'bias' in layer:
        continue
    iters = data['iterations']
    rownorms = data['row_norms']
    colnorms = data['col_norms']
    average_row_norm = torch.mean(rownorms,dim=0)
    average_col_norm = torch.mean(colnorms,dim=0)
    aggregate_stats[layer] = {
        'average_row_norm': average_row_norm,
        'average_col_norm': average_col_norm
    }

#print(aggregate_stats)
layer_names = list(aggregate_stats.keys())

# Stack row and column norms across layers
row_norms_all = torch.stack(
    [aggregate_stats[name]['average_row_norm'] for name in layer_names]
)  # shape: (L, R)

col_norms_all = torch.stack(
    [aggregate_stats[name]['average_col_norm'] for name in layer_names]
)  # shape: (L, C)


def top_x_percent(norms_2d: torch.Tensor, percent: float):
    """
    norms_2d: tensor of shape (num_layers, dim)
    percent: 0-1 (fraction) or 0-100 (percentage)
    returns:
        values: (k,)
        layer_idx: (k,) indices into 0..num_layers-1
        inner_idx: (k,) row/col index within that layer
    """
    L, D = norms_2d.shape
    flat = norms_2d.view(-1)  # (L*D,)

    total = flat.numel()
    if percent <= 1:
        k = max(1, int(total * percent))
    else:
        k = max(1, int(total * percent / 100.0))
    k = min(k, total)

    values, flat_idx = torch.topk(flat, k, largest=True, sorted=True)
    layer_idx = flat_idx // D
    inner_idx = flat_idx % D
    return values, layer_idx, inner_idx

def top_x_percent_equal(norms_2d: torch.Tensor, percent: float):
    """
    norms_2d: tensor of shape (num_layers, dim)
    percent: 0-1 (fraction) or 0-100 (percentage)
    returns:
        values: (L * k_per_layer,)
        layer_idx: (L * k_per_layer,) indices into 0..num_layers-1
        inner_idx: (L * k_per_layer,) row/col index within that layer
    """
    L, D = norms_2d.shape

    # how many per layer
    if percent <= 1:
        k_per_layer = max(1, int(D * percent))
    else:
        k_per_layer = max(1, int(D * percent / 100.0))
    k_per_layer = min(k_per_layer, D)

    values_list = []
    layer_idx_list = []
    inner_idx_list = []

    for l in range(L):
        row = norms_2d[l]                              # shape: (D,)
        vals, idx = torch.topk(row, k_per_layer, largest=True, sorted=True)
        values_list.append(vals)
        inner_idx_list.append(idx)
        layer_idx_list.append(
            torch.full(
                (k_per_layer,),
                l,
                dtype=torch.long,
                device=norms_2d.device,
            )
        )

    # concat over layers
    values = torch.cat(values_list, dim=0)        # (L * k_per_layer,)
    layer_idx = torch.cat(layer_idx_list, dim=0)  # (L * k_per_layer,)
    inner_idx = torch.cat(inner_idx_list, dim=0)  # (L * k_per_layer,)

    # optional: globally sort the selected entries by value descending
    values, order = torch.sort(values, descending=True)
    layer_idx = layer_idx[order]
    inner_idx = inner_idx[order]

    return values, layer_idx, inner_idx

# Top x% by row and by column (across all layers)
if equal_between_layers:
    row_vals, row_layer_idx, row_row_idx = top_x_percent_equal(row_norms_all, top_percent) 
    col_vals, col_layer_idx, col_col_idx = top_x_percent_equal(col_norms_all, top_percent)
else:
    row_vals, row_layer_idx, row_row_idx = top_x_percent(row_norms_all, top_percent)
    col_vals, col_layer_idx, col_col_idx = top_x_percent(col_norms_all, top_percent)

# Optional: build human-readable lists with layer names + indices
top_rows = [
    {
        "layer": layer_names[int(li)],
        "row_index": int(ri),
        "value": float(v),
    }
    for v, li, ri in zip(row_vals, row_layer_idx, row_row_idx)
]

top_cols = [
    {
        "layer": layer_names[int(li)],
        "col_index": int(ci),
        "value": float(v),
    }
    for v, li, ci in zip(col_vals, col_layer_idx, col_col_idx)
]

print("Num top rows:", len(top_rows))
print("Num top cols:", len(top_cols))
print("Example top row entries:", top_rows[:5])
print("Example top col entries:", top_cols[:5])

layer_counts = Counter(entry["layer"] for entry in top_cols)
pprint(dict(layer_counts))

preselect_grads = {
    'percent': top_percent * 100,
    'layers': {}
}
 
for layer in layer_names:
    biases_to_train = []
    if train_all_bias:
        biases_to_train = torch.arange(shape[1]).tolist()
    preselect_grads['layers'][roberta_layer_to_preselect_name(layer)] = {
        'shape': shape,
        'train_weight_indices': [],
        'train_bias_indices': biases_to_train
    }
for entry in top_cols:
    layer = entry["layer"]
    col_idx = entry["col_index"]
    preselect_name = roberta_layer_to_preselect_name(layer)
    preselect_grads['layers'][preselect_name]['train_weight_indices'].extend(
        [[r, col_idx] for r in range(shape[0])]
    )

with open(f"masks/{mask_name}", "w") as f:
    json.dump(preselect_grads, f)
