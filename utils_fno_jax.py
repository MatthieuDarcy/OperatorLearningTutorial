import jax.numpy as jnp


#%%

from typing import Sequence, Optional, Callable 
from flax import linen as nn
import jax
import jax.numpy as jnp

#%%


class MLP(nn.Module):
    out_channels: int
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, x):
        h = self.out_channels
        x = nn.Dense(h)(x)
        x = self.activation(x)
        x = nn.Dense(h)(x)
        return x
    

class SpectralConv2d(nn.Module):
    width: int
    modes: int
    @nn.compact
    def __call__(self, x):
        batchsize, height, width, channels = x.shape
        w_f = width//2 + 1 # number of Fourier modes for rfft 
        m1 = min(self.modes, height) # number of Fourier modes for rfft
        m2 = min(self.modes, w_f) # number of Fourier modes for rfft

        w_pos = self.param(
            "w_pos",
            lambda k, s: jax.random.normal(k, s, dtype=jnp.complex64) * (1.0 / self.width),
            (m1, m2, self.width, self.width),
        )
        w_neg = self.param(
            "w_neg",
            lambda k, s: jax.random.normal(k, s, dtype=jnp.complex64) * (1.0 / self.width),
            (m1, m2, self.width, self.width),
        )
        # FFT -> apply learned complex weights to low modes -> iFFT
        x_ft = jnp.fft.rfft2(x, axes=(1, 2)) # shape (batchsize, height, width//2 + 1, channels)

        out_ft = jnp.zeros((batchsize, height, w_f, self.width), dtype=jnp.complex64)

        out_pos = jnp.sum(x_ft[:, :m1, :m2, :, None]*w_pos[None], axis = -2)
        out_neg = jnp.sum(x_ft[:, -m1:, :m2, :, None]*w_neg[None], axis = -2)

        out_ft = out_ft.at[:, :m1, :m2, :].set(out_pos)
        out_ft = out_ft.at[:, -m1:, :m2, :].set(out_neg)

        x = jnp.fft.irfft2(out_ft, s=(height, width), axes=(1, 2))

        return x

class FNOBlock2d(nn.Module):
    width: int
    modes: int
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, x):
        y_spec = SpectralConv2d(width=self.width, modes=self.modes)(x)
        y_local = nn.Dense(self.width)(x)  # pointwise mixing
        return self.activation(y_spec + y_local)

class FNO2d(nn.Module):
    modes: int = 12
    width: int = 64
    in_channels: int = 1
    out_channels: int = 1
    n_layers: int = 4
    activation: Callable = nn.gelu
    lift_hidden: Optional[int] = None

    @nn.compact
    def __call__(self, x, *, train: bool = False):
        
        # lift operator
        x = MLP(self.width)(x) # automatically applies to the last dimension (pointwise)
        #print("mlp out ", x)
        for _ in range(self.n_layers):
            x = x + FNOBlock2d(width=self.width, modes=self.modes, activation=self.activation)(x)
            #print("fno block out ", x)
        # Projection to output space
        x = MLP(self.out_channels)(x)
        #print("final out ", x)
        return x

        