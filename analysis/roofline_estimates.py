import argparse
import math

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mem_bw_gbps", type=float, default=600.0, help="Assumed device memory bandwidth GB/s")
    ap.add_argument("--bytes_per_token", type=float, default=200e6, help="Rough bytes moved per token (KV-heavy decode).")
    ap.add_argument("--measured_ms_per_token", type=float, default=5.0, help="Measured per-token latency in ms.")
    args = ap.parse_args()

    bw_bytes_s = args.mem_bw_gbps * 1e9
    ideal_s = args.bytes_per_token / bw_bytes_s
    ideal_ms = ideal_s * 1e3

    achieved_s = args.measured_ms_per_token / 1e3
    achieved_bw = args.bytes_per_token / achieved_s / 1e9  # GB/s

    print("=== Roofline-style sanity check ===")
    print(f"Assumed BW: {args.mem_bw_gbps:.1f} GB/s")
    print(f"Bytes per token (estimate): {args.bytes_per_token:.2e} bytes")
    print(f"Ideal latency if purely bandwidth-limited: {ideal_ms:.3f} ms/token")
    print(f"Measured latency: {args.measured_ms_per_token:.3f} ms/token")
    print(f"Implied achieved bandwidth: {achieved_bw:.1f} GB/s")
    print("")
    print("Interpretation:")
    print("- If implied achieved BW is near device BW -> likely memory-bound (KV/attention).")
    print("- If far lower -> overheads, poor locality, launch/sync, or compute bottlenecks.")

if __name__ == "__main__":
    main()