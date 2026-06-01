from flax import linen as nn

import logging
import jax
import jax.numpy as jnp

import itertools
import functools

from typing import Tuple, Callable, List, Optional, Iterable, Any
from flax.struct import dataclass
from evojax.task.base import TaskState
from evojax.policy.base import PolicyNetwork
from evojax.policy.base import PolicyState
from evojax.util import create_logger
from evojax.util import get_params_format_fn

import jax
import jax.numpy as jnp
from flax import linen as nn

class SafeConv3x3(nn.Module):
    features: int
    padding: str = "VALID"  # "VALID" pour réduire la taille, "SAME" pour du zero-padding

    @nn.compact
    def __call__(self, x):
        # x a une forme (H, W, C)
        H, W, C = x.shape
        neighborhood = []
        
        if self.padding == "VALID":
            # On extrait les 9 voisins en réduisant la taille de l'image à (H-2, W-2)
            for dy in range(3):
                for dx in range(3):
                    slice_x = x[dy : dy + H - 2, dx : dx + W - 2, :]
                    neighborhood.append(slice_x)
                    
        elif self.padding == "SAME":
            # On applique un zero-padding de 1 pixel sur les bords spatiaux
            padded_x = jnp.pad(x, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
            # On extrait les 9 voisins en conservant la taille initiale (H, W)
            for dy in range(3):
                for dx in range(3):
                    slice_x = padded_x[dy : dy + H, dx : dx + W, :]
                    neighborhood.append(slice_x)
        else:
            raise ValueError(f"Padding type {self.padding} non supporté.")
            
        flat_neighbors = jnp.concatenate(neighborhood, axis=-1)
        out = nn.Dense(self.features)(flat_neighbors)
        return out
    
    
class MetaRNN_bcppr(nn.Module):
    output_size: int
    out_fn: str
    hidden_layers: list
    encoder_in: bool
    encoder_layers: list

    def setup(self):
        self._num_micro_ticks = 1
        self._lstm = nn.recurrent.LSTMCell(features=4)
        
        # Remplacement des nn.Conv de Flax par vos SafeConv stables sur GPU
        self.conv1 = SafeConv3x3(features=4, padding="VALID")
        self.conv2 = SafeConv3x3(features=8, padding="VALID")

        self._hiddens = [(nn.Dense(size)) for size in self.hidden_layers]
        self._output_proj = nn.Dense(self.output_size)
        if self.encoder_in:
            self._encoder = [(nn.Dense(size)) for size in self.encoder_layers]

    def __call__(self, h, c, inputs: jnp.ndarray, last_action: jnp.ndarray, reward: jnp.ndarray):
        carry = (h, c)
        
        # inputs est de forme (7, 7, 2)
        out = inputs
        
        
        # Passage dans le premier bloc convolutif safe
        out = self.conv1(out)
        out = nn.relu(out)
        # avg_pool attend un tenseur avec une dimension de batch ou s'applique sur les axes spatiaux.
        # Pour (7, 7, 4), on spécifie les dimensions de la fenêtre locale.
        out = nn.avg_pool(out, window_shape=(2, 2), strides=(2, 2), padding="SAME")
        
        # Passage dans le second bloc convolutif safe
        out = self.conv2(out)
        out = nn.relu(out)
        out = nn.avg_pool(out, window_shape=(2, 2), strides=(2, 2), padding="SAME")
        
        # Aplatissement propre du tenseur de caractéristiques spatiales extrait
        out = out.reshape(-1)

        if self.encoder_in and len(self._encoder) > 0:
            for layer in self._encoder:
                out = jax.nn.tanh(layer(out))

        inputs_encoded = jnp.concatenate([out, last_action, reward])

        for _ in range(self._num_micro_ticks):
            carry, out = self._lstm(carry, inputs_encoded)
            
        out = jnp.concatenate([inputs_encoded, out])
        for layer in self._hiddens:
            out = jax.nn.tanh(layer(out))
        out = self._output_proj(out)

        h, c = carry
        if self.out_fn == 'tanh':
            out = nn.tanh(out)
        elif self.out_fn == 'softmax':
            out = nn.softmax(out, axis=-1)
        else:
            if self.out_fn != 'categorical':
                raise ValueError('Unsupported output activation: {}'.format(self.out_fn))
        return h, c, out

@dataclass
class metaRNNPolicyState_bcppr(PolicyState):
    lstm_h: jnp.array
    lstm_c: jnp.array
    keys: jnp.array


class MetaRnnPolicy_bcppr(PolicyNetwork):

    def __init__(self, input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 output_act_fn: str = "categorical",
                 hidden_layers: list = [],
                 encoder: bool = False,
                 encoder_layers: list = [32, 32],
                 logger: logging.Logger = None):

        if logger is None:
            self._logger = create_logger(name='MetaRNNolicy')
        else:
            self._logger = logger
        model = MetaRNN_bcppr(output_dim, out_fn=output_act_fn, hidden_layers=hidden_layers, encoder_in=encoder,
                              encoder_layers=encoder_layers)
        self.params = model.init(jax.random.PRNGKey(0), jnp.zeros((hidden_dim)), jnp.zeros((hidden_dim)),
                                 jnp.zeros(input_dim), jnp.zeros([output_dim]), jnp.zeros([1]))

        self.num_params, format_params_fn = get_params_format_fn(self.params)
        self._logger.info('MetaRNNPolicy.num_params = {}'.format(self.num_params))
        self.hidden_dim = hidden_dim
        self._format_params_fn = jax.jit(jax.vmap(format_params_fn))
        self._forward_fn = jax.jit(jax.vmap(model.apply))

    def reset(self, states: TaskState) -> PolicyState:
        """Reset the policy.
        Args:
            TaskState - Initial observations.
        Returns:
            PolicyState. Policy internal states.
        """
        keys = jax.random.split(jax.random.PRNGKey(0), states.obs.shape[0])
        h = jnp.zeros((states.obs.shape[0], self.hidden_dim))
        c = jnp.zeros((states.obs.shape[0], self.hidden_dim))
        return metaRNNPolicyState_bcppr(keys=keys, lstm_h=h, lstm_c=c)

    def reset_b(self, obs: jnp.array) -> PolicyState:
        """Reset the policy.
        Args:
            TaskState - Initial observations.
        Returns:
            PolicyState. Policy internal states.
        """
        keys = jax.random.split(jax.random.PRNGKey(0), obs.shape[0])
        h = jnp.zeros((obs.shape[0], self.hidden_dim))
        c = jnp.zeros((obs.shape[0], self.hidden_dim))
        return metaRNNPolicyState_bcppr(keys=keys, lstm_h=h, lstm_c=c)

    def get_actions(self, t_states: TaskState, params: jnp.ndarray, p_states: PolicyState):
        params = self._format_params_fn(params)
        h, c, out = self._forward_fn(params, p_states.lstm_h, p_states.lstm_c, t_states.obs, t_states.last_actions,
                                     t_states.rewards)
        return out, metaRNNPolicyState_bcppr(keys=p_states.keys, lstm_h=h, lstm_c=c)