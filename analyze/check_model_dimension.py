import argparse
from transformers import AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Check dimensions of OPT model layers. "
            "You can provide either --model_id (default: facebook/opt-6.7b) or --checkpoint_dir to load a local checkpoint directory."
        )
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/opt-6.7b",
        help="Model ID to load the OPT model from (e.g. 'facebook/opt-6.7b'). Ignored if --checkpoint_dir is provided.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Path to a local checkpoint directory (e.g. one saved by FSDP finetuning). If provided, loads model from here instead of --model_id.",
    )
    args = parser.parse_args()

    if args.checkpoint_dir is not None:
        model = AutoModelForCausalLM.from_pretrained(args.checkpoint_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_id)
    # The OPT model's decoder layers are in model.model.decoder.layers
    layers = model.model.decoder.layers
    num_layers = len(layers)
    print(f"Number of decoder layers: {num_layers}")

    # Get hidden size and ffn dimension from the first layer
    hidden_size = layers[0].self_attn.q_proj.out_features
    ffn_dim = layers[0].fc1.out_features
    print(f"Hidden size: {hidden_size}")
    print(f"FFN dimension: {ffn_dim}")

    for i, layer in enumerate(layers):
        print(f"\nLayer {i}:")
        # Attention projection matrices
        q_proj_weight = layer.self_attn.q_proj.weight
        k_proj_weight = layer.self_attn.k_proj.weight
        v_proj_weight = layer.self_attn.v_proj.weight
        out_proj_weight = layer.self_attn.out_proj.weight

        print(f"  q_proj weight shape: {q_proj_weight.shape} (rows x cols)")
        print(f"  k_proj weight shape: {k_proj_weight.shape} (rows x cols)")
        print(f"  v_proj weight shape: {v_proj_weight.shape} (rows x cols)")
        print(f"  out_proj weight shape: {out_proj_weight.shape} (rows x cols)")

        # MLP layers
        fc1_weight = layer.fc1.weight
        fc2_weight = layer.fc2.weight
        print(f"  fc1 weight shape: {fc1_weight.shape} (rows x cols)")
        print(f"  fc2 weight shape: {fc2_weight.shape} (rows x cols)")

    # Summary info
    print("\nSummary:")
    print(f"Number of layers: {num_layers}")
    print(f"Hidden size: {hidden_size}")
    print(f"FFN dimension: {ffn_dim}")
    print(f"Number of neurons (fc1 out_features): {ffn_dim}")

if __name__ == "__main__":
    main()
