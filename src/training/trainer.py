import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

from config.config import Config


def train_temporal_gnn(
    model,
    train_sequences,
    train_targets,
    edge_index,
    val_sequences=None,
    val_targets=None,
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    patience=Config.PATIENCE,
    device=Config.DEVICE,
):
    """
    Train temporal GNN. Uses mixed-precision acceleration on CUDA; falls back
    to plain fp32 on CPU (autocast on CPU breaks oneDNN's LSTM kernel and gives
    no speedup anyway since CPUs lack tensor cores).

    Learns baseline creek physics via reconstruction (MSE loss). After training,
    computes a spill-detection threshold from validation-set errors so that
    inference uses a stable, training-defined definition of "anomalous" rather
    than recomputing percentiles on whatever it happens to see at inference time.

    Returns:
        train_losses: list of per-epoch training loss
        val_losses:   list of per-epoch validation loss (empty if no val data)
        threshold:    float, P_THRESHOLD_PERCENTILE of validation errors,
                      or None if no validation data was provided
    """
    model = model.to(device)
    edge_index = edge_index.to(device)

    device_type = "cuda" if "cuda" in str(device) else "cpu"
    use_amp = device_type == "cuda"

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    scaler = GradScaler(device_type) if use_amp else None

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    train_losses = []
    val_losses = []

    print(f"--- Starting Training on {device_type.upper()} "
          f"({'mixed-precision' if use_amp else 'fp32'}) ---")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0

        # Mini-batch training
        for i in range(0, len(train_sequences), batch_size):
            batch_seq = torch.FloatTensor(train_sequences[i:i+batch_size]).to(device)
            batch_target = torch.FloatTensor(train_targets[i:i+batch_size]).to(device)

            optimizer.zero_grad()

            if use_amp:
                with autocast(device_type=device_type):
                    predictions = model(
                        batch_seq,
                        edge_index,
                        batch_size=len(batch_seq),
                        num_nodes=batch_seq.shape[2],
                    )
                    loss = criterion(predictions, batch_target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                predictions = model(
                    batch_seq,
                    edge_index,
                    batch_size=len(batch_seq),
                    num_nodes=batch_seq.shape[2],
                )
                loss = criterion(predictions, batch_target)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)

        # Validation
        if val_sequences is not None:
            model.eval()
            with torch.no_grad():
                val_seq = torch.FloatTensor(val_sequences).to(device)
                val_tgt = torch.FloatTensor(val_targets).to(device)
                if use_amp:
                    with autocast(device_type=device_type):
                        val_pred = model(
                            val_seq,
                            edge_index,
                            batch_size=len(val_seq),
                            num_nodes=val_seq.shape[2],
                        )
                        val_loss = criterion(val_pred, val_tgt).item()
                else:
                    val_pred = model(
                        val_seq,
                        edge_index,
                        batch_size=len(val_seq),
                        num_nodes=val_seq.shape[2],
                    )
                    val_loss = criterion(val_pred, val_tgt).item()
            val_losses.append(val_loss)

            # Track best model and handle early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[STATUS] Early stopping triggered at epoch {epoch+1}")
                    break

        # Progress logging every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            status = f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {avg_train_loss:.6f}"
            if val_sequences is not None:
                status += f" | Val Loss: {val_loss:.6f}"
            print(status)

    # Restore best weights if available
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"[INFO] Restored model weights from epoch with Val Loss: {best_val_loss:.6f}")

    # ─── Compute the spill threshold from validation errors ────────────────
    # We use the validation set (held-out, never trained on) to define what
    # "normal" prediction error looks like. Anything beyond the Nth percentile
    # of validation errors is considered anomalous at inference time.
    #
    # This is computed once during training and saved to model metadata.
    # Critically: it does NOT recompute at inference time, which is what
    # makes the threshold a stable, meaningful definition of anomalous
    # rather than "the top 1% of whatever data we happened to see this run."
    threshold = None
    if val_sequences is not None:
        model.eval()
        with torch.no_grad():
            val_seq = torch.FloatTensor(val_sequences).to(device)
            val_tgt = torch.FloatTensor(val_targets).to(device)
            val_pred = model(
                val_seq,
                edge_index,
                batch_size=len(val_seq),
                num_nodes=val_seq.shape[2],
            )
            val_errors = (val_pred - val_tgt).abs().cpu().numpy()
            # Match the system-level aggregation used in anomaly_detector:
            # mean across nodes and features then one score per timestep.
            system_scores = val_errors.mean(axis=(1, 2))
            threshold = float(
                np.percentile(system_scores, Config.THRESHOLD_PERCENTILE)
            )
        print(
            f"[INFO] Computed spill threshold from validation set: "
            f"{threshold:.6f} (P{Config.THRESHOLD_PERCENTILE})"
        )

    print("--- Training Complete ---\n")
    return train_losses, val_losses, threshold
