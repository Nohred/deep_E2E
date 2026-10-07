import matplotlib.pyplot as plt
import numpy as np

# history = {
#         "train_loss": [],
#         "val_loss": [],
#         "learning_rate": []
#     }

def plot_segmentation(
    images,
    masks,
    predictions,
):
    num_images = len(images)

    fig, axes = plt.subplots(
        3,
        num_images,
        figsize=(4 * num_images, 10),
    )

    if num_images == 1:
        axes = axes.reshape(3, 1)

    cmap = plt.get_cmap(
        "tab20",
        21,
    ).copy()

    cmap.set_bad("gray")

    for i in range(num_images):
        image = images[i].permute(
            1,
            2,
            0,
        )

        axes[0, i].imshow(image)
        axes[0, i].axis("off")

        mask = masks[i].cpu().numpy()
        mask = np.ma.masked_where(
            mask == 255,
            mask,
        )

        axes[1, i].imshow(
            mask,
            cmap=cmap,
            vmin=0,
            vmax=20,
        )

        axes[1, i].axis("off")

        axes[2, i].imshow(
            predictions[i].cpu(),
            cmap=cmap,
            vmin=0,
            vmax=20,
        )

        axes[2, i].axis("off")

    axes[0, 0].set_ylabel("Imagen")
    axes[1, 0].set_ylabel("Real")
    axes[2, 0].set_ylabel("Predicción")

    plt.tight_layout()
    plt.show()
       

def plot_history( history,
                ):
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax, lr_ax) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    ax.plot(
        epochs,
        history['train_loss'],
        label='Training loss',
        color='#2563eb',
        linewidth=2.5,
        marker='o',
        markersize=4,
    )

    if 'val_loss' in history:
        ax.plot(
            epochs,
            history['val_loss'],
            label='Validation loss',
            color='#dc2626',
            linewidth=2.5,
            marker='o',
            markersize=4,
        )

    ax.set_title('Model Loss', fontsize=16, fontweight='bold', pad=12)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_xticks(list(epochs))
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False)
    ax.set_facecolor('#f8fafc')


    lr_ax.plot(
        epochs,
        history['learning_rate'],
        label='Learning rate',
        color='#16a34a',
        linewidth=2.5,
        marker='o',
        markersize=4,
    )
    lr_ax.set_yscale('log')  # Set y-axis to logarithmic scale for better visualization
    lr_ax.set_title('Learning Rate', fontsize=16, fontweight='bold', pad=12)
    lr_ax.set_xlabel('Epoch')
    lr_ax.set_ylabel('Learning rate')
    lr_ax.set_xticks(list(epochs))
    lr_ax.grid(True, linestyle='--', alpha=0.3)
    lr_ax.spines['top'].set_visible(False)
    lr_ax.spines['right'].set_visible(False)
    lr_ax.legend(frameon=False)
    lr_ax.set_facecolor('#f8fafc')

    fig.patch.set_facecolor('white')

    #show the plot
    plt.show()
    return fig, ax

    