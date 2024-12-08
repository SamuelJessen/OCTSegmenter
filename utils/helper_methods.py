import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from matplotlib.colors import ListedColormap

def plot_cv_indices(cv, X, y, ax, n_splits, lw=10, cmap_data="tab10"):
    """Create a sample plot for indices of a cross-validation object."""

    # Define a custom colormap excluding #ff7f0e
    original_colors = plt.cm.tab10.colors  # Default tab10 colormap colors
    filtered_colors = [color for color in original_colors if color != (255/255, 127/255, 14/255)]  # Remove #ff7f0e
    custom_cmap = ListedColormap(filtered_colors)

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1  # Mark the test samples
        indices[tr] = 0  # Mark the train samples

        # Visualize the results for the current split
        # Train samples in light blue, validation samples in redder orange
        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=np.where(indices == 0, '#add8e6', '#ff7f0e'),  # Set light blue and redder orange
            marker="_",
            lw=lw,
        )

    # Plot the unique_id at the end (instead of class labels)
    ax.scatter(
        range(len(X)), [ii + 1.5] * len(X), c=y, marker="_", lw=lw, cmap=custom_cmap
    )

    # Add a legend for train and validation splits
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#add8e6', lw=4, label='Train'),
        Line2D([0], [0], color='#ff7f0e', lw=4, label='Validation'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

    # Formatting
    yticklabels = list(range(n_splits)) + ["scanning"]
    ax.set(
        yticks=np.arange(n_splits + 1) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[n_splits + 1.2, -0.2],
        xlim=[0, len(X)],
    )
    ax.set_title(f"{type(cv).__name__} Cross-Validation", fontsize=12)
    return ax

# Visualize splits
def visualize_cv_splits(metadata_df, n_splits=9):
    # Extract unique IDs and their corresponding target (unique_id)
    unique_ids = metadata_df["unique_id"].values

    # Initialize StratifiedKFold with n_splits
    skf = StratifiedKFold(n_splits=n_splits)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the cross-validation splits
    plot_cv_indices(
        skf, X=metadata_df, y=unique_ids, ax=ax, n_splits=n_splits
    )

    plt.show()

# Custom function to shorten trial directory names
def trial_dirname_creator(trial):
    # Shorten the trial name to only include key parameters
    return f"trial_{trial.trial_id}_lr={trial.config['lr']:.1e}_opt={trial.config['optimizer']}_bs={trial.config['batch_size']}_model={trial.config['model']}_freeze={trial.config['freeze_encoder']}_loss={trial.config['loss_function']}_fold={trial.config['fold']}"


def plot_cv_indices_only_one_fold(cv, X, y, ax, lw=10, cmap_data="tab10"):
    """Create a sample plot for indices of the first fold of a cross-validation object."""
    # Define a custom colormap excluding #ff7f0e
    original_colors = plt.cm.tab10.colors  # Default tab10 colormap colors
    filtered_colors = [color for color in original_colors if color != (255/255, 127/255, 14/255)]  # Remove #ff7f0e
    custom_cmap = ListedColormap(filtered_colors)

    # Generate the training/testing visualizations for the first CV split
    tr, tt = next(cv.split(X=X, y=y))
    
    # Fill in indices with the training/test groups
    indices = np.array([np.nan] * len(X))
    indices[tt] = 1  # Mark the test samples
    indices[tr] = 0  # Mark the train samples

    # Visualize the results for the first split
    # Train samples in light blue, validation samples in redder orange
    ax.scatter(
        range(len(indices)),
        [0.5] * len(indices),
        c=np.where(indices == 0, '#add8e6', '#ff7f0e'),  # Set light blue and redder orange
        marker="_",
        lw=lw,
    )

    # Plot the unique_id at the end (instead of class labels)
    ax.scatter(
        range(len(X)), [1.0] * len(X), c=y, marker="_", lw=lw, cmap=custom_cmap
    )

    # Add a legend for train and validation splits
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#add8e6', lw=4, label='Train'),
        Line2D([0], [0], color='#ff7f0e', lw=4, label='Validation'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

    # Formatting
    yticklabels = ["Fold 1", "scanning"]
    ax.set(
        yticks=[0.5, 1.0],
        yticklabels=yticklabels,
        xlabel="Sample index",
        #ylabel="CV iteration",
        ylim=[1.5, -0.5],
        xlim=[0, len(X)],
    )
    ax.set_title(f"{type(cv).__name__} (First Fold)", fontsize=10)
    return ax

# Visualize splits
def visualize_cv_splits_only_one_fold(metadata_df, n_splits=9):
    # Extract unique IDs and their corresponding target (unique_id)
    unique_ids = metadata_df["unique_id"].values

    # Initialize StratifiedKFold with n_splits
    skf = StratifiedKFold(n_splits=n_splits)

    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 2.5))

    # Plot the cross-validation splits
    plot_cv_indices_only_one_fold(
        skf, X=metadata_df, y=unique_ids, ax=ax
    )

    plt.show()