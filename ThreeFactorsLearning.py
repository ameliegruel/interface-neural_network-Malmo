from abc import ABC
from typing import Union, Optional, Sequence

import torch
import numpy as np

from bindsnet.network.topology import (
    AbstractConnection,
    Connection,
    Conv2dConnection,
    LocalConnection,
)

class STDP(ABC):
    # language=rst
    """
    Three factors learning rule : 1) STDP with eligibility trace  2) dopamine reward  3) weight threshold leading to learning interruption 
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        tc_eligibility_trace: float = 0.0,
        tc_reward: float = 0.0,
        tc_minus: float =0.0,
        tc_plus: float = 0.0,
        **kwargs
    ) -> None:
        # language=rst
        """
        Abstract constructor for the ``LearningRule`` object.

        :param connection: An ``AbstractConnection`` object whose weights the ``PostPre`` learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events ().
        :param reduction: Method for reducing parameter updates along the batch dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.
        :param tc_eligibility_trace: Time constant for the eligibility trace.
        :param tc_reward: Time constant for the reward.
        :param tc_minus: Time constant for post-synaptic firing trace.
        :param tc_plus: Time constant for pre-synaptic firing trace.
        """
        # Connection parameters.
        self.connection = connection
        self.source = connection.source
        self.target = connection.target

        self.wmin = connection.wmin
        self.wmax = connection.wmax

        # Learning rate(s).
        if nu is None:
            nu = [0.0, 0.0]
        elif isinstance(nu, float) or isinstance(nu, int):
            nu = [nu, nu]

        self.nu = torch.tensor(nu)

        # Initialize eligibility trace, time constants and reward
        if not hasattr(self.target, "eligibility_trace"):
            self.target.eligibility_trace = torch.zeros(*self.connection.w.shape) 
        self.tc_eligibility_trace = tc_eligibility_trace
        self.tc_minus = torch.tensor(tc_minus)
        self.tc_plus = torch.tensor(tc_plus)
        if not hasattr(self.connection, "reward_concentration"):
            self.connection.reward_concentration = torch.zeros(*self.connection.w.shape) # initialize the extracellular concentration of biogenic amine
        self.tc_reward = tc_reward

        # Parameter update reduction across minibatch dimension.
        if reduction is None:
            reduction = torch.mean
        self.reduction = reduction

        # Weight decay.
        self.weight_decay = weight_decay

        assert (
            self.source.traces and self.target.traces
        ), "Both pre- and post-synaptic nodes must record spike traces."

        if isinstance(connection, (Connection, LocalConnection, Conv2dConnection)) == False:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )

    def update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule method.
        """
        if not hasattr(self, "reward"):
            self.reward = kwargs["reward"]   # amount of biogenic amine released 
                                    # NB : argument 'reward' is defined in Network() initialization, not in STDP()

        batch_size = self.source.batch_size

        pre_x = self.source.x.view(-1).unsqueeze(1)
        post_x = self.target.x.view(-1).unsqueeze(1)

        # Get STDP
        delta_t = post_x - pre_x
        tau = torch.where(delta_t > 0, -self.tc_plus, self.tc_minus)
        nu = torch.where(delta_t > 0, self.nu[1], -self.nu[0])
        STDP = nu * torch.exp(delta_t / tau)
        # self.connection.w += STDP

        # Update eligibility trace
        update = -self.target.eligibility_trace / self.tc_eligibility_trace + STDP 
        self.target.eligibility_trace += self.connection.dt * update

        # Update reward
        update = -self.connection.reward_concentration / self.tc_reward
        # self.connection.reward_concentration += update * self.connection.dt + self.reward
        self.connection.reward_concentration += update * self.connection.dt

        # Update weight
        update = self.target.eligibility_trace * self.connection.reward_concentration * self.connection.dt
        self.connection.w = torch.nn.Parameter(torch.max(torch.tensor(0.0001), self.connection.w + update))

        # Implement weight decay
        if self.weight_decay:
            self.connection.w -= self.weight_decay * self.connection.w

        # Bound weights.
        if (self.connection.wmin != -np.inf or self.connection.wmax != np.inf):
            self.connection.w.clamp_(self.connection.wmin, self.connection.wmax)

