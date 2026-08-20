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

import jax
import jax.numpy as jnp
import flax.linen as nn


class SafeConv(nn.Module):
    features: int
    kernel_size: tuple = (3, 3)
    strides: tuple = (1, 1)
    padding: str = "SAME"          # comme le defaut de nn.Conv
    use_bias: bool = True

    @nn.compact
    def __call__(self, x):
        # x : (H, W, C)
        H, W, C = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.strides

        if self.padding == "SAME":
            out_h = -(-H // sh)            # ceil(H / sh)
            out_w = -(-W // sw)
            pad_h = max(0, (out_h - 1) * sh + kh - H)
            pad_w = max(0, (out_w - 1) * sw + kw - W)
            pt, pb = pad_h // 2, pad_h - pad_h // 2   # excedent cote haut/droite
            pl, pr = pad_w // 2, pad_w - pad_w // 2
            xpad = jnp.pad(x, ((pt, pb), (pl, pr), (0, 0)))
        elif self.padding == "VALID":
            out_h = (H - kh) // sh + 1
            out_w = (W - kw) // sw + 1
            xpad = x
        else:
            raise ValueError(f"padding {self.padding} non supporte")

        # Un slice sous-echantillonne par decalage (dy, dx) du noyau.
        # Chacun a la forme (out_h, out_w, C) ; on les concatene -> (.., .., kh*kw*C).
        patches = []
        for dy in range(kh):
            for dx in range(kw):
                patches.append(
                    xpad[dy:dy + (out_h - 1) * sh + 1:sh,
                         dx:dx + (out_w - 1) * sw + 1:sw, :]
                )
        patch = jnp.concatenate(patches, axis=-1)
        return nn.Dense(self.features, use_bias=self.use_bias)(patch)


class MetaRNN_bcppr(nn.Module):

    output_size: int
    out_fn: str
    hidden_layers: list
    encoder_in: bool
    encoder_layers: list
    carry_size: int = 4          # doit valoir hidden_dim : reset_b dimensionne h/c dessus
    # "separee" : cablage actuel, la memoire ne voit pas la vision.
    # "jointe"  : cablage d'avant 591269d, conserve pour pouvoir le rejouer et
    #             le comparer. Voir __call__ pour ce que chacun change.
    memory_mode: str = "separee"

    def setup(self):
        self._num_micro_ticks = 1
        self._lstm = nn.recurrent.LSTMCell(features=self.carry_size)
        self.conv1 = SafeConv(features=4, kernel_size=(3, 3),
                              strides=(2, 2), padding="SAME")
        self.conv2 = SafeConv(features=8, kernel_size=(3, 3),
                              strides=(2, 2), padding="SAME")
        self._hiddens = [nn.Dense(size) for size in self.hidden_layers]
        self._output_proj = nn.Dense(self.output_size)
        if self.encoder_in:
            self._encoder = [nn.Dense(size) for size in self.encoder_layers]

    def __call__(self, h, c, inputs: jnp.ndarray,
                 last_action: jnp.ndarray, reward: jnp.ndarray, energy: jnp.ndarray,
                 last_eaten: jnp.ndarray):
        carry = (h, c)

        out = inputs
        out = self.conv1(out)
        out = nn.relu(out)
        out = nn.max_pool(out, window_shape=(2, 2), strides=(1, 1))  # VALID par defaut

        out = self.conv2(out)
        out = nn.relu(out)
        out = nn.max_pool(out, window_shape=(2, 2), strides=(1, 1))

        out = jnp.ravel(out)

        if self.encoder_in:
            for layer in self._encoder:
                out = jax.nn.tanh(layer(out))

        if self.memory_mode == "separee":
            # Deux voies separees.
            #
            # La MEMOIRE ne recoit pas la vision : seulement ce que l'agent vient
            # de manger, ce que ca lui a rapporte, et son etat interne.
            # `last_eaten` et `reward` portent tous deux sur le pas precedent,
            # donc l'association a retenir -- ce canal vaut tant -- arrive d'un
            # bloc, sans qu'il y ait d'assignation de credit a faire a travers le
            # temps.
            mem_in = jnp.concatenate([last_action, reward, energy, last_eaten])
            for _ in range(self._num_micro_ticks):
                carry, mem = self._lstm(carry, mem_in)

            # La TETE combine la scene courante, qui suffit a naviguer, et la
            # valeur apprise des canaux, qui n'existe que dans le carry. Pour
            # eviter le poison il faut donc s'en servir : l'observation seule ne
            # dit pas quel canal est toxique.
            out = jnp.concatenate([out, last_action, reward, energy, mem])

        elif self.memory_mode == "jointe":
            # Cablage d'avant 591269d, repris a l'identique.
            #
            # Le LSTM recoit la vision encodee, et sa sortie est reinjectee A
            # COTE de cette meme vision : 4 dimensions sur 42 a l'entree de la
            # tete. Le chemin reactif domine, et ablater le carry ne change
            # presque rien -- c'est ce constat qui a motive la separation.
            #
            # `last_eaten` n'est PAS concatene : il n'existait pas dans ce
            # reseau. L'ignorer est ce qui redonne exactement le nombre de
            # parametres et le comportement du run d'origine.
            inputs_encoded = jnp.concatenate([out, last_action, reward, energy])
            for _ in range(self._num_micro_ticks):
                carry, mem = self._lstm(carry, inputs_encoded)
            out = jnp.concatenate([inputs_encoded, mem])

        else:
            raise ValueError(
                f"memory_mode inconnu : {self.memory_mode!r} "
                '(attendu "separee" ou "jointe")')
        for layer in self._hiddens:
            out = jax.nn.tanh(layer(out))
        out = self._output_proj(out)

        h, c = carry
        if self.out_fn == 'tanh':
            out = nn.tanh(out)
        elif self.out_fn == 'softmax':
            out = nn.softmax(out, axis=-1)
        elif self.out_fn != 'categorical':
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
                 memory_mode: str = "separee",
                 logger: logging.Logger = None):

        if logger is None:
            self._logger = create_logger(name='MetaRNNolicy')
        else:
            self._logger = logger
        model = MetaRNN_bcppr(output_dim, out_fn=output_act_fn, hidden_layers=hidden_layers, encoder_in=encoder,
                              encoder_layers=encoder_layers, carry_size=hidden_dim,
                              memory_mode=memory_mode)
        # input_dim = (cote, cote, n_types + agents + murs) -> on en deduit n_types
        # plutot que de l'ajouter au constructeur : impossible de le desynchroniser.
        n_types = input_dim[2] - 2
        self.params = model.init(jax.random.PRNGKey(0), jnp.zeros((hidden_dim)), jnp.zeros((hidden_dim)),
                                 jnp.zeros(input_dim), jnp.zeros([output_dim]), jnp.zeros([1]), jnp.zeros([1]),
                                 jnp.zeros([n_types]))

        self.num_params, format_params_fn = get_params_format_fn(self.params)
        self._logger.info('MetaRNNPolicy.num_params = {}'.format(self.num_params))
        self.hidden_dim = hidden_dim
        self.memory_mode = memory_mode
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
                                     t_states.rewards,
                                     jnp.expand_dims(t_states.agents.energy, 1).astype(jnp.float32),
                                     t_states.last_eaten.astype(jnp.float32))
        return out, metaRNNPolicyState_bcppr(keys=p_states.keys, lstm_h=h, lstm_c=c)