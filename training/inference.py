"""
Single-flow / single-batch inference entrypoint used for latency benchmarking
across deployment tiers (manuscript Section 6.3, Table 4).
"""
import argparse
import time
import torch


def benchmark_latency(model, sample_input, device, n_warmup=20, n_runs=200):
    model.eval()
    sample_input = [t.to(device) for t in sample_input] if isinstance(sample_input, (list, tuple)) else sample_input.to(device)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(*sample_input) if isinstance(sample_input, (list, tuple)) else model(sample_input)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_runs):
            model(*sample_input) if isinstance(sample_input, (list, tuple)) else model(sample_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

    avg_latency_ms = ((end - start) / n_runs) * 1000
    return avg_latency_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    raise NotImplementedError("Load model + representative sample input, then call benchmark_latency().")