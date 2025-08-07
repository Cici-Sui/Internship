#第三版 画loss随epoch变化的图像
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main(csv_path, bin_size=5):
    # Load metrics CSV
    df = pd.read_csv(csv_path)

    # Extract training losses per batch (logged each batch)
    df_train = df[~df['train/loss'].isna()][['epoch', 'train/loss']].copy()

    # Prepare containers for plotting
    train_epoch_x = []            # x-values for epoch-level training loss
    train_epoch_loss = []         # y-values for epoch-level training loss
    train_mid_x = []              # x-values for midpoints between epochs
    train_avg_batches = []        # y-values for averaged training loss per bin

    # Group by epoch
    for epoch, group in df_train.groupby('epoch'):
        # Record the final training loss at the end of the epoch
        epoch_loss = group['train/loss'].iloc[-1]
        train_epoch_x.append(epoch)
        train_epoch_loss.append(epoch_loss)

        # Compute averages per bin_size batches
        total_batches = len(group)
        num_groups = total_batches // bin_size
        for g in range(num_groups):
            start_idx = g * bin_size
            end_idx = start_idx + bin_size
            avg_loss = group['train/loss'].iloc[start_idx:end_idx].mean()
            # Place mid‑point evenly between epoch and epoch+1
            x_val = epoch + (g + 0.5) / num_groups
            train_mid_x.append(x_val)
            train_avg_batches.append(avg_loss)

    # Extract validation losses (logged once per epoch)
    df_valid = df[~df['valid/loss'].isna()][['epoch', 'valid/loss']]
    # Ensure one entry per epoch
    df_valid = df_valid.drop_duplicates(subset='epoch')
    valid_x = df_valid['epoch'].tolist()
    valid_loss = df_valid['valid/loss'].tolist()

    # Plotting
    plt.figure(figsize=(8, 5))
    # Training: averaged per bin
    plt.plot(train_mid_x, train_avg_batches, '-', label=f'Train Loss (avg/{bin_size} batches)')
    # Training: epoch-level
    plt.plot(train_epoch_x, train_epoch_loss, 'o-', label='Train Loss per Epoch')
    # Validation: epoch-level
    plt.plot(valid_x, valid_loss, 's-', label='Validation Loss per Epoch')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save figure
    out_path = 'loss_curve.png'
    plt.savefig(out_path)
    plt.close()
    print(f"Saved loss curve to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot training and validation loss curves')
    parser.add_argument('csv_path', help='Path to metrics CSV file')
    parser.add_argument('--bin_size', type=int, default=5,
                        help='Number of batches to average for intermediate points')
    args = parser.parse_args()
    main(args.csv_path, bin_size=args.bin_size)


#第二版不要了

# 第一版 loss是散点图， r的图不对
# import sys
# import pandas as pd
# import matplotlib.pyplot as plt

# def main(csv_path):
#     df = pd.read_csv(csv_path)

#     # 1) Loss curves (batch-level logs get aggregated to epoch‐mean)
#     plt.figure(figsize=(6,4))
#     plt.plot(df["epoch"], df["train/loss"],  marker="o", label="Train Loss")
#     plt.plot(df["epoch"], df["valid/loss"],  marker="o", label="Valid Loss")
#     plt.xlabel("Epoch"), plt.ylabel("Loss")
#     plt.title("LFADS Loss vs. Epoch")
#     plt.legend(), plt.grid(True), plt.tight_layout()
#     plt.savefig("loss_curve.png")

#     # 2) Pearson’s r (only logged once per epoch, so no duplicates)
#     df_r = df[["epoch", "train/pearson_r_epoch", "valid/pearson_r_epoch"]] \
#              .drop_duplicates(subset="epoch")
#     plt.figure(figsize=(6,4))
#     plt.plot(df_r["epoch"],
#              df_r["train/pearson_r_epoch"],  marker="s", label="Train Pearson r")
#     plt.plot(df_r["epoch"],
#              df_r["valid/pearson_r_epoch"],  marker="s", label="Valid Pearson r")
#     plt.xlabel("Epoch"), plt.ylabel("Pearson r")
#     plt.title("LFADS Pearson r vs. Epoch")
#     plt.ylim(0,1), plt.legend(), plt.grid(True), plt.tight_layout()
#     plt.savefig("pearson_r_curve.png")

#     print("Saved: loss_curve.png, pearson_r_curve.png")

# if __name__=="__main__":
#     if len(sys.argv)!=2:
#         print("Usage: python plot_metrics.py path/to/metrics.csv")
#     else:
#         main(sys.argv[1])
