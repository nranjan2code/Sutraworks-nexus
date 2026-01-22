# NEXUS-Reason-Alpha: Launch Manual

**Objective:** Provision GCP resources and launch training for the 450M Reasoning Model.
**Estimated Time:** 15 Minutes.
**Budget Safety:** Auto-shutdown at $250.00.

---

## Phase 1: Infrastructure Provisioning (GCP Console)

1.  **Navigate to Compute Engine > VM Instances**.
2.  **Click "Create Instance"**.
3.  **Configuration**:
    *   **Name:** `nexus-alpha-trainer`
    *   **Region:** `us-central1` or `us-east1` (Cheapest/Most availability).
    *   **Zone:** Any (e.g., `us-central1-a`).
    *   **Machine Configuration:** `GPU` -> `NVIDIA L4` (1 GPU).
    *   **Machine Type:** `g2-standard-4` (4 vCPU, 16GB Memory).
    *   **VM Provisioning Model:** **Spot** (Critical for budget!).
    *   **Boot Disk:** Switch to **Deep Learning on Linux** -> `Debian 11 based Deep Learning VM M115` (Torch 2.4, CUDA 12.1).
    *   **Disk Size:** **200 GB Balanced Persistent Disk**.
4.  **Firewall:** Check "Allow HTTP/HTTPS" (useful for WandB/Jupyter if needed).
5.  **Click Create**.

---

## Phase 2: Code Deployment

1.  **SSH into the VM**:
    ```bash
    gcloud compute ssh nexus-alpha-trainer
    ```

2.  **Clone Your Repository**:
    ```bash
    git clone https://github.com/nranjan2code/Sutraworks-nexus
    cd Sutraworks-nexus
    ```

3.  **Run the Setup Script**:
    ```bash
    # This installs PyTorch, dependencies, and sets up venv
    bash scripts/gcp_startup.sh
    ```

---

## Phase 3: Launch Training

1.  **Activate Environment**:
    ```bash
    source venv/bin/activate
    ```

2.  **Establish Persistence (Tmux)**:
    *   We use `tmux` so training doesn't die if your internet disconnects.
    ```bash
    tmux new -s training
    ```

3.  **Login to WandB (Optional but Recommended)**:
    ```bash
    wandb login
    # Paste your API key
    ```

4.  **IGNITION**:
    ```bash
    python scripts/train_alpha.py
    ```

---

## Phase 4: Monitoring & Verification

1.  **Verify Loss**:
    *   Look for the `[Budget]` log lines every 100 steps.
    *   Ensure `loss` is decreasing (starts around 10.0, should drop below 4.0 quickly).

2.  **Detach**:
    *   Press `Ctrl+B`, then `D` to detach from tmux.
    *   You can now safely close your SSH terminal.

3.  **Re-attach**:
    ```bash
    gcloud compute ssh nexus-alpha-trainer
    tmux attach -t training
    ```

---

## Emergency Abort
If you need to stop everything immediately to save money:
```bash
# In the VM
pkill python

# From your local machine (Nuclear option)
gcloud compute instances delete nexus-alpha-trainer
```
