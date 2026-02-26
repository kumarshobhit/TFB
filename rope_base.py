import argparse
import math


def _validate_rope_args(f_dom, d_model, n_heads):
    try:
        freq = float(f_dom)
    except (TypeError, ValueError):
        raise ValueError("f_dom must be a numeric value.")

    if freq <= 0:
        raise ValueError("f_dom must be > 0.")

    if not isinstance(d_model, int) or not isinstance(n_heads, int):
        raise ValueError("d_model and n_heads must be integers.")

    if d_model <= 0 or n_heads <= 0:
        raise ValueError("d_model and n_heads must be > 0.")

    if d_model % n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads.")

    d_head = d_model // n_heads
    return freq, d_head


def _valid_k_range(d_head):
    max_k = d_head // 2 - 1
    if max_k < 1:
        return []
    return list(range(1, max_k + 1))


def compute_rope_base_list(f_dom, d_model=128, n_heads=8):
    """
    Compute base_k values for all valid RoPE frequency bands.

    Returns:
        list[dict]: [{"k": k, "base_k": value}, ...]
    """
    freq, d_head = _validate_rope_args(f_dom, d_model, n_heads)
    k_values = _valid_k_range(d_head)
    if not k_values:
        return []

    c_val = 1.0 / (2.0 * math.pi * freq)
    out = []
    for k in k_values:
        base_k = c_val ** (d_head / (2.0 * k))
        out.append({"k": int(k), "base_k": float(base_k)})
    return out


def get_rope_base_for_k(f_dom, k, d_model=128, n_heads=8):
    """
    Compute a single base_k value for a specified band index k.
    """
    freq, d_head = _validate_rope_args(f_dom, d_model, n_heads)
    k_values = _valid_k_range(d_head)
    if not k_values:
        raise ValueError(
            f"No valid k values exist for d_head={d_head}. "
            "Need d_head >= 4 to have at least one valid k."
        )

    try:
        k_int = int(k)
    except (TypeError, ValueError):
        raise ValueError("k must be an integer.")

    if k_int not in k_values:
        raise ValueError(f"k must be in [{k_values[0]}, {k_values[-1]}], got {k_int}.")

    c_val = 1.0 / (2.0 * math.pi * freq)
    return float(c_val ** (d_head / (2.0 * k_int)))


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Compute RoPE base_k values from input frequency."
    )
    parser.add_argument("--freq", type=float, required=True, help="Frequency in cycles/step.")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension.")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Optional single band index. If omitted, prints full valid k range.",
    )
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.k is None:
        base_list = compute_rope_base_list(
            f_dom=args.freq, d_model=args.d_model, n_heads=args.n_heads
        )
        for item in base_list:
            print(f"k={item['k']} base_k={item['base_k']}")
    else:
        value = get_rope_base_for_k(
            f_dom=args.freq, k=args.k, d_model=args.d_model, n_heads=args.n_heads
        )
        print(f"k={args.k} base_k={value}")


if __name__ == "__main__":
    main()
