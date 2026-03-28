import argparse
import math


def _validate_sine_args(f_dom, d_model):
    try:
        freq = float(f_dom)
    except (TypeError, ValueError):
        raise ValueError("f_dom must be a numeric value.")

    if freq <= 0:
        raise ValueError("f_dom must be > 0.")

    if not isinstance(d_model, int):
        raise ValueError("d_model must be an integer.")

    if d_model <= 0:
        raise ValueError("d_model must be > 0.")

    return freq, d_model


def _pair_count(d_model):
    return len(range(0, d_model, 2))


def _valid_k_range(d_model):
    pair_count = _pair_count(d_model)
    if pair_count <= 1:
        return []
    return list(range(1, pair_count))


def compute_sine_base_list(f_dom, d_model=128):
    """
    Compute base_k values for all valid sinusoidal frequency pairs.

    Standard sinusoidal positional encoding uses one angular frequency per
    even/odd channel pair, not one frequency per raw feature dimension. If:

        div_term[k] = base^(-2k / d_model)

    and the dominant data frequency f_dom is given in cycles/step, then its
    angular frequency is:

        omega_dom = 2 * pi * f_dom

    Matching one sinusoidal PE pair k to the dominant angular frequency gives:

        omega_dom = div_term[k]

    Solving for base yields:

        base_k = (1 / (2 * pi * f_dom))^(d_model / (2 * k))

    Notes:
    - The relevant pair index k is over channel pairs created by
      torch.arange(0, d_model, 2), so the valid raw pair range is
      [0, pair_count - 1].
    - k=0 is excluded for base derivation because div_term[0] = 1 for any base,
      so it cannot determine a unique base.

    Returns:
        list[dict]: [{"k": k, "base_k": value}, ...]
    """
    freq, d_model = _validate_sine_args(f_dom, d_model)
    k_values = _valid_k_range(d_model)
    if not k_values:
        return []

    c_val = 1.0 / (2.0 * math.pi * freq)
    out = []
    for k in k_values:
        base_k = c_val ** (d_model / (2.0 * k))
        out.append({"k": int(k), "base_k": float(base_k)})
    return out


def get_sine_base_for_k(f_dom, k, d_model=128):
    """
    Compute a single base_k value for a specified sinusoidal pair index k.
    """
    freq, d_model = _validate_sine_args(f_dom, d_model)
    k_values = _valid_k_range(d_model)
    if not k_values:
        raise ValueError(
            f"No valid k values exist for d_model={d_model}. "
            "Need at least two even/odd pairs to solve for a base."
        )

    try:
        k_int = int(k)
    except (TypeError, ValueError):
        raise ValueError("k must be an integer.")

    if k_int not in k_values:
        raise ValueError(f"k must be in [{k_values[0]}, {k_values[-1]}], got {k_int}.")

    c_val = 1.0 / (2.0 * math.pi * freq)
    return float(c_val ** (d_model / (2.0 * k_int)))


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Compute sinusoidal PE base_k values from dominant frequency."
    )
    parser.add_argument("--freq", type=float, required=True, help="Frequency in cycles/step.")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension.")
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Optional single pair index. If omitted, prints full valid k range.",
    )
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.k is None:
        base_list = compute_sine_base_list(f_dom=args.freq, d_model=args.d_model)
        for item in base_list:
            print(f"k={item['k']} base_k={item['base_k']}")
    else:
        value = get_sine_base_for_k(f_dom=args.freq, k=args.k, d_model=args.d_model)
        print(f"k={args.k} base_k={value}")


if __name__ == "__main__":
    main()
