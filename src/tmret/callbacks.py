from IPython import display
from lightning.pytorch.callbacks import Callback
import matplotlib.pyplot as plt


class CleanDisplay(Callback):

    def __init__(self, every_n_epoch: int = 1):
        super().__init__()
        self.every_n_epoch = every_n_epoch

    def on_train_epoch_end(self, trainer, pl_module):
        # Get metrics at the end of each epoch
        current_epoch = trainer.current_epoch
        if (current_epoch) % self.every_n_epoch == 0:
            display.clear_output(wait=True)

            # Re-enable the progress bar
            if hasattr(trainer, "progress_bar_callback"):
                trainer.progress_bar_callback.on_train_epoch_end(trainer, pl_module)


class PlotLossCallback(Callback):
    _training_color: str = 'purple'
    _train_color: str = 'blue'
    _valid_color: str = 'red'
    _learning_rate_color: str = 'green'
    _marker_size: int = 7
    _line_width: int = 2

    def __init__(self, skip_validation: bool = False):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        self.lr = []
        self.skip_validation = skip_validation
        plt.ion()  # Turn on interactive mode

    def on_train_epoch_end(self, trainer, pl_module):
        # Get metrics at the end of each epoch
        current_epoch = trainer.current_epoch
        self.epochs.append(current_epoch + 1)
        train_loss = trainer.callback_metrics.get("train_loss", None)
        val_loss = trainer.callback_metrics.get("val_loss", None)
        current_lr = trainer.optimizers[0].param_groups[0]['lr']

        if train_loss is not None:
            self.train_losses.append(train_loss.item())
        if not self.skip_validation:
            if val_loss is not None:
                self.val_losses.append(val_loss.item())
        if current_lr is not None:
            self.lr.append(current_lr)

        self._update_plot()

    def _update_plot(self):
        """Update the plot in real-time"""

        plt.clf()  # Clear the previous plot
        plt.figure(figsize=(8, 4))

        ax1 = plt.gca()
        ax1.grid(ls=':')
        ax1.plot(self.epochs, self.train_losses, label="Train loss", color=self._train_color, ms=self._marker_size,
                 marker='.')
        if not self.skip_validation:
            ax1.plot(self.epochs, self.val_losses, label="Valid Loss", color=self._valid_color, ms=self._marker_size,
                     marker='.')
        ax1.set_yscale('log')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss", color=self._training_color)
        ax1.tick_params(axis='y', which='both', labelcolor=self._training_color, width=2)
        ax1.spines['left'].set_color('purple')  # Axis color
        ax1.spines['left'].set_linewidth(2)  # Axis line thickness

        ax2 = ax1.twinx()
        ax2.plot(self.epochs, self.lr, label="Learning rate", color=self._learning_rate_color, ms=self._marker_size,
                 marker='.')
        ax2.set_ylabel("Learning rate", color=self._learning_rate_color)
        ax2.set_yscale('log')
        ax2.tick_params(axis='y', which='both', labelcolor=self._learning_rate_color, width=2)
        ax2.spines['right'].set_color(self._learning_rate_color)  # Axis color
        ax2.spines['right'].set_linewidth(2)  # Axis line thickness

        # Increase linewidth for top and bottom axes
        ax1.spines['top'].set_linewidth(2)
        ax1.spines['bottom'].set_linewidth(2)
        ax1.tick_params(axis='x', which='both', width=2)

        # Create a combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        # Title and layout + display
        plt.title("Training dynamics")
        plt.tight_layout()
        plt.show()