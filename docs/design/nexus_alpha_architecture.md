# NEXUS-Reason-Alpha: Edge Architecture (Pi 5 Edition)

## 1. Executive Summary: The "Edge-First" Mandate

**Target Hardware:** Raspberry Pi 5 (4GB / 8GB versions).
**Constraint:** CPU-bound inference (ARM Cortex-A76).
**Goal:** A "System 2" Reasoning SML that runs locally > 15 t/s.

We have refactored the design from a "Server-Small" model to an **"Edge-Native"** model.

---

## 2. Architecture: "Deep & Narrow"

To run fast on a CPU, we must minimize the dimensionality of matrix multiplications (`d_model`). A wide model (1024/2048) chokes the CPU cache. A narrow model (768/512) fits in L2/L3 cache better and executes faster.

We switch from `1024x24` to **`768x32`**:
*   **Faster Inference:** `768^2` is 56% the cost of `1024^2`.
*   **Deeper Reasoning:** 32 layers provide more "thinking steps" for the reasoning head.
*   **SSM Advantage:** The Hybrid SSM architecture uses a fixed-size state, meaning **RAM usage does not grow** with sequence length. This is critical for 4GB Pi devices.

### 2.1 Final Specifications

| Property | Value | Notes |
| :--- | :--- | :--- |
| **Parameters** | **~380 Million** | Fits in ~750MB RAM (FP16) or ~250MB (Int4). |
| **d_model** | 768 | Optimized for ARM NEON vectorization. |
| **Layers** | 32 | Deep reasoning capabilities. |
| **Vocab** | 32,000 | Standard Llama tokenizer (High compatibility). |
| **Context** | Infinite | Thanks to SSM recurrence. |

---

## 3. Raspberry Pi 5 Performance Model

### 3.1 Memory Budget (4GB Model)
*   **OS/System:** 1.0 GB
*   **Display/Shared:** 0.5 GB
*   **Usable for AI:** ~2.5 GB
*   **NEXUS-Alpha Requirement:**
    *   FP32: 1.52 GB (Fits)
    *   FP16: 0.76 GB (Comfortable)
    *   Int4: 0.25 GB (Trivial)

**Verdict:** The model is extremely safe for the 4GB Pi 5. You could even run **three** of them simultaneously.

### 3.2 Throughput Estimate
*   **Pi 5 Compute:** ~10-15 GFLOPS/core (sustained python/numpy).
*   **NEXUS-Alpha Cost:** ~0.7 GFLOPs per token.
*   **Est Speed:** **~15 - 25 tokens/second** (Unquantized).
*   **Est Speed (Int8/4):** **30+ tokens/second**.

This makes the model "Chattable" in real-time.

---

## 4. Financial & Training Plan (Unchanged)

The training budget remains **$300**.
*   The architecture change (Narrower) actually **speeds up training** slightly on the L4 GPU.
*   We will produce the same "smart" model, just shaped for the Edge.

### 4.1 Deployment Pipeline
1.  **Train** on GCP (L4 Spot).
2.  **Export** to `.onnx` or GGUF.
3.  **Run** on Pi 5 using `llama.cpp` or NEXUS-Edge runtime (Python).

---

## 5. Revised Risk Analysis
*   **Pi 5 Thermals:** Continuous inference at 25 t/s will throttle the Pi 5. **Active Cooling (Fan) is mandatory.**
*   **Quantization:** While 380M fits in FP16, quantization to Int8 is recommended to reduce memory bandwidth pressure on the Pi's shared RAM.
