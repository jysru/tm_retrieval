import torch
from torch import optim, nn
import torch.nn.functional as F
import lightning as L


class LitTransmissionMatrix(L.LightningModule):
    _default_optimizer = torch.optim.Adam
    _default_optimizer_kwargs: dict = dict(
        lr=1e-3,
    )
    _default_init_lr: float = 1e-3
    _default_scheduler = optim.lr_scheduler.ReduceLROnPlateau
    _default_scheduler_kwargs: dict = dict(
        factor=0.5,
        patience=10,
        threshold=0.01,
        cooldown=3,
        min_lr=1e-7,
    )

    def __init__(
            self,
            input_size: int,
            output_size: int,
            train_loss_fn=nn.MSELoss(),
            init_lr: float = None,
            optimizer=None,
            optimizer_kwargs: dict = None,
            scheduler=None,
            scheduler_kwargs: dict = None
    ):
        super().__init__()
        self.save_hyperparameters()
        self._init_lr = init_lr if init_lr is not None else LitTransmissionMatrix._default_init_lr
        self._optimizer = optimizer if optimizer is not None else LitTransmissionMatrix._default_optimizer
        self._optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else LitTransmissionMatrix._default_optimizer_kwargs
        self._scheduler = scheduler if scheduler is not None else LitTransmissionMatrix._default_scheduler
        self._scheduler_kwargs = scheduler_kwargs if scheduler_kwargs is not None else LitTransmissionMatrix._default_scheduler_kwargs
        self._input_size = input_size
        self._output_size = output_size

        layers = [
            nn.Linear(input_size, output_size, bias=False, dtype=torch.cdouble),
        ]

        # Create sequential model
        self.model = nn.Sequential(*layers)

        # Define loss function
        self.loss_fn = train_loss_fn

    def forward(self, x, with_abs=True):
        if with_abs:
            return torch.abs(self.model(x))
        else:
            return self.model(x)

    def training_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = self._optimizer(self.parameters(), **self._optimizer_kwargs)
        lr_scheduler = self._scheduler(optimizer, **self._scheduler_kwargs)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss",
            }
        }



class LitTMFourierCorrection(L.LightningModule):
    _default_optimizer = torch.optim.Adam
    _default_optimizer_kwargs: dict = dict(
        lr=1e-3,
    )
    _default_init_lr: float = 1e-3
    _default_scheduler = optim.lr_scheduler.ReduceLROnPlateau
    _default_scheduler_kwargs: dict = dict(
        factor=0.5,
        patience=10,
        threshold=0.01,
        cooldown=3,
        min_lr=1e-7,
    )

    def __init__(
            self,
            tm_model: LitTransmissionMatrix,
            pad_to: int,
            train_loss_fn=nn.MSELoss(),
            init_lr: float = None,
            optimizer=None,
            optimizer_kwargs: dict = None,
            scheduler=None,
            scheduler_kwargs: dict = None
    ):
        super().__init__()
        self.save_hyperparameters()
        self._init_lr = init_lr if init_lr is not None else LitTMFourierCorrection._default_init_lr
        self._optimizer = optimizer if optimizer is not None else LitTMFourierCorrection._default_optimizer
        self._optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else LitTMFourierCorrection._default_optimizer_kwargs
        self._scheduler = scheduler if scheduler is not None else LitTMFourierCorrection._default_scheduler
        self._scheduler_kwargs = scheduler_kwargs if scheduler_kwargs is not None else LitTMFourierCorrection._default_scheduler_kwargs

        # Freeze the tm_model parameters
        self.tm_model = tm_model
        self.tm_model.eval()
        for param in self.tm_model.parameters():
            param.requires_grad = False

        self._pad_to = pad_to
        self._img_size = int(torch.sqrt(torch.tensor((self.tm_model._output_size))))
        self._pad_n = int((self._pad_to - self._img_size) // 2)

        # Initialize phase_bias as a learnable parameter
        init_phase = 2 * torch.pi * torch.rand((1, self.tm_model._output_size))
        self.phase_bias = nn.Parameter(init_phase.clone())

        self.loss_fn = train_loss_fn

    def forward(self, x):
        y = self.tm_model.forward(x, with_abs=False) # Size is N x m², type cdouble
        y = y * torch.exp(1j * self.phase_bias)
        y = y.view(y.shape[0], 1, self._img_size, self._img_size) # Size is now N x 1 x m x m, type cdouble
        y = F.pad(y, (self._pad_n, self._pad_n, self._pad_n, self._pad_n), "constant", 0)
        print(y.shape)
        z = torch.abs(self.fourier_transform(y))
        return z

    @staticmethod
    def fourier_transform(field: torch.Tensor, norm: str = 'ortho') -> torch.Tensor:
        """
        Compute the 2D Fourier transform of complex-valued input images in [N, C, H, W],
        applying FFT over the last two dims (H, W), with fftshift before and after.
        Returns the normalized FFT result.
        """
        if not torch.is_complex(field):
            raise ValueError("Input must be a complex-valued tensor.")

        # Apply fftshift before FFT
        field = torch.fft.fftshift(field, dim=(-2, -1))

        # Compute 2D FFT over H and W
        ft = torch.fft.fftn(field, dim=(-2, -1), norm=norm)

        # Apply fftshift after FFT
        ft = torch.fft.fftshift(ft, dim=(-2, -1))

        return ft

    def training_step(self, batch):
        x, z = batch
        z_hat = self(x)
        z = z.view(z.shape[0], 1, self._img_size, self._img_size) # Size is now N x 1 x m x m, type double
        z = F.pad(z, (self._pad_n, self._pad_n, self._pad_n, self._pad_n), "constant", 0)
        loss = self.loss_fn(z_hat, z)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch):
        x, z = batch
        z_hat = self(x)
        z = z.view(z.shape[0], 1, self._img_size, self._img_size)  # Size is now N x 1 x m x m, type double
        z = F.pad(z, (self._pad_n, self._pad_n, self._pad_n, self._pad_n), "constant", 0)
        loss = self.loss_fn(z_hat, z)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = self._optimizer(self.parameters(), **self._optimizer_kwargs)
        lr_scheduler = self._scheduler(optimizer, **self._scheduler_kwargs)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "train_loss",
            }
        }
