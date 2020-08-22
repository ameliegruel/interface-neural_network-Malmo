from abc import ABC
from typing import Union, Optional, Sequence, Iterable

import torch
from torch.nn import Module, Parameter
# torch.set_default_tensor_type("torch.cuda.FloatTensor")

import numpy as np
from bindsnet.learning.learning import NoOp
from bindsnet.network.nodes import Nodes
from bindsnet.network.topology import (
    AbstractConnection,
    Connection,
    Conv2dConnection,
    LocalConnection,
)


class Izhikevich(Nodes):
    # language=rst
    """
    Layer of Izhikevich neurons.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        traces: bool = False,
        traces_additive: bool = False,
        tc_trace: Union[float, torch.Tensor] = 20.0,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        sum_input: bool = False,
        excitatory: float = 1,
        thresh: Union[float, torch.Tensor] = 45.0,
        rest: Union[float, torch.Tensor] = -65.0,
        lbound: float = None,
        a=0.01,
        b=0.2,
        c=-65,
        d=8,
        C=4,
        k=0.035,
        noise_mean=0,
        noise_std=0.05,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a layer of Izhikevich neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param excitatory: Percent of excitatory (vs. inhibitory) neurons in the layer; in range ``[0, 1]``.
        :param thresh: Spike threshold voltage.
        :param rest: Resting membrane voltage.
        :param lbound: Lower bound of the voltage.
        """
        super().__init__(
            n=n,
            shape=shape,
            traces=traces,
            traces_additive=traces_additive,
            tc_trace=tc_trace,
            trace_scale=trace_scale,
            sum_input=sum_input,
        )

        self.register_buffer("rest", torch.tensor(rest))  # Rest voltage.
        self.register_buffer("thresh", torch.tensor(thresh))  # Spike threshold voltage.

        # set parameters
        self.a = a
        self.b = b
        self.c = torch.tensor(c).float()
        self.d = d
        self.C = C
        self.k = k

        self.register_buffer("v", self.rest * torch.ones(n))  # Neuron voltages.
        self.register_buffer("u", torch.zeros(n))  # Neuron recovery.

        self.noise_mean = noise_mean
        self.noise_std = noise_std

        # spiking times
        self.t_spike = -10000*torch.ones(self.n)
        self.t = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Runs a single simulation step.
        :param x: Inputs to the layer.
        """

        self.t += 1

        # Check for spiking neurons.
        self.s = self.v >= self.thresh

        # Voltage and recovery reset.
        self.v = torch.where(self.s, self.c, self.v)
        self.u = torch.where(self.s, self.u + self.d, self.u)

        # set new spike time
        self.t_spike = torch.where(self.s, self.t, self.t_spike)

        # Add noise
        noise = self.noise_mean + self.noise_std * torch.normal(0,1,([self.n]))

        # Apply v and u updates.
        self.v += self.dt * 0.5 * ((self.k * (self.v - self.rest) * (self.v - self.thresh) - self.u + x + noise) / self.C)
        self.v += self.dt * 0.5 * ((self.k * (self.v - self.rest) * (self.v - self.thresh) - self.u + x + noise) / self.C)
        self.u += self.dt * self.a * (self.b * (self.v - self.rest) - self.u)

        super().forward(x)

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_state_variables()
        self.v.fill_(self.rest)  # Neuron voltages.
        self.u = self.b * self.v  # Neuron recovery.

    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)
        self.v = self.rest * torch.ones(batch_size, *self.shape, device=self.v.device)
        self.u = self.b * self.v


######################################################################


class AllToAllConnection(ABC, Module):

    def __init__(
        self,
        source: Nodes,
        target: Nodes,
        w: None,
        tc_synaptic: float = 0.0,
        phi: float = 0.0,
        nu: Optional[Union[float, Sequence[float]]] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        """
        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param nu: Learning rate for both pre- and post-synaptic events.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param LearningRule update_rule: Modifies connection parameters according to
            some rule.
        :param torch.Tensor w: Strengths of synapses.
        :param torch.Tensor b: Target population bias
        :param float wmin: The minimum value on the connection weights.
        :param float wmax: The maximum value on the connection weights.
        :param float norm: Total weight per target neuron normalization.
        """
        super().__init__() # initialisation of Module

        assert isinstance(source, Nodes), "Source is not a Nodes object"
        assert isinstance(target, Nodes), "Target is not a Nodes object"

        self.source = source
        self.target = target

        self.nu = nu
        self.weight_decay = weight_decay
        
        self.update_rule = kwargs.get("update_rule", NoOp)
        self.wmin = kwargs.get("wmin", -np.inf)
        self.wmax = kwargs.get("wmax", np.inf)
        self.norm = kwargs.get("norm", None)
        
        # Learning rule
        if self.update_rule is None:
            self.update_rule = NoOp

        self.update_rule = self.update_rule(
            connection=self,
            nu=nu,
            weight_decay=weight_decay,
            **kwargs
        )

        # Weights
        self.w = Parameter(w, requires_grad=False)
        self.b = Parameter(kwargs.get("b", torch.zeros(target.n)), requires_grad=False)

        # Parameters used to update synaptic input
        self.active_neurotransmitters = torch.zeros(self.source.n, self.target.n)
        self.tc_synaptic = tc_synaptic
        self.phi = phi
        self.v_rev = 0

        self.cumul_I = None
        self.cumul_weigth = self.w.t()
        if not hasattr(self.target, "eligibility_trace"):
            self.target.eligibility_trace = torch.zeros(*self.w.shape)
        self.cumul_et = self.target.eligibility_trace.t()

    # Get dirac(delta_t)
    def get_dirac(self) : 
        pre_s  = self.source.s.view(-1).unsqueeze(1)
        post_s = self.target.s
        return torch.max(pre_s, post_s).float()  # True or 1 if a spike occured either in pre or post neuron, False or 0 otherwise


    def compute(self, s: torch.Tensor) -> None:
        # language=rst
        """
        Compute pre-activations of downstream neurons given spikes of upstream neurons.

        :param s: Incoming spikes.
        """

        # Update of the number of active neurotransmitters for each synapse
        pre_spike_occured = torch.mul(s.float().view(-1,1), torch.ones(*self.active_neurotransmitters.shape))
        update = - self.active_neurotransmitters / self.tc_synaptic + self.phi * pre_spike_occured
        update = torch.where(self.w != 0, update, torch.tensor(0.))
        self.active_neurotransmitters += update

        # Get input
        S = torch.sum(self.active_neurotransmitters.t(), dim=1, keepdim=True).view(1,-1)
        return (self.v_rev - self.target.v) * torch.max(self.w) * S
        # if self.cumul_I == None:
        #     self.cumul_I = I
        # else :
        #     self.cumul_I = torch.cat((self.cumul_I, I),0)
        # return I

    def update(self, **kwargs) -> None:
        # language=rst
        """
        Compute connection's update rule.

        Keyword arguments:

        :param bool learning: Whether to allow connection updates.
        :param ByteTensor mask: Boolean mask determining which weights to clamp to zero.
        """
        learning = kwargs.get("learning", True)

        self.cumul_weigth = torch.cat((self.cumul_weigth, self.w.t()),0)
        self.cumul_et = torch.cat((self.cumul_et,self.target.eligibility_trace.t()),0)

        if learning:
            self.update_rule.update(**kwargs)

        mask = kwargs.get("mask", None)
        if mask is not None:
            self.w.masked_fill_(mask, 0)

    def normalize(self) -> None:
        """
        Normalize weights so each target neuron has sum of connection weights equal to
        ``self.norm``.
        """
        if self.norm is not None:
            w_abs_sum = self.w.abs().sum(0).unsqueeze(0)
            w_abs_sum[w_abs_sum == 0] = 1.0
            self.w *= self.norm / w_abs_sum

    def reset_state_variables(self) -> None:
        """
        Contains resetting logic for the connection.
        """
        pass


######################################################


class STDP(ABC):
    # language=rst
    """
    Three factors learning rule : 1) STDP with eligibility trace  2) dopamine reward  3) weight threshold leading to learning interruption
    """

    def __init__(
        self,
        connection: AllToAllConnection,
        nu: Optional[Union[float, Sequence[float]]] = None,
        weight_decay: float = 0.0,
        tc_eligibility_trace: float = 0.0,
        tc_reward: float = 0.0,
        reward: float = 0.0,
        tc_minus: float = 0.0,
        tc_plus: float = 0.0,
        min_weight: float = 0.0,
        **kwargs
    ) -> None:
        """
        Abstract constructor for the ``LearningRule`` object.

        :param connection: An ``AbstractConnection`` object whose weights the ``PostPre`` learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events ().
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
        self.t = 0 # time in ms, number of updates

        self.wmin = connection.wmin
        self.wmax = connection.wmax

        # Learning rate(s).
        if nu is None:
            nu = [0.0, 0.0]
        elif isinstance(nu, float) or isinstance(nu, int):
            nu = [nu, nu]

        self.nu = torch.tensor(nu)

        # Weight threshold
        self.min_weight = min_weight

        # Initialize eligibility trace, time constants and reward
        if not hasattr(self.target, "eligibility_trace"):
            self.target.eligibility_trace = torch.zeros(*self.connection.w.shape)
        self.tc_eligibility_trace = tc_eligibility_trace
        self.tc_minus = torch.tensor(tc_minus)
        self.tc_plus = torch.tensor(tc_plus)
        if not hasattr(self.connection, "reward_concentration"):
            self.connection.reward_concentration = torch.zeros(*self.connection.w.shape) # initialize the extracellular concentration of biogenic amine
        self.tc_reward = tc_reward
        self.BA = reward # nul for every t except t=40 ms

        # Weight decay.
        self.weight_decay = weight_decay

        assert (
            self.source.traces and self.target.traces
        ), "Both pre- and post-synaptic nodes must record spike traces."

        if isinstance(connection, (Connection, LocalConnection, Conv2dConnection, AllToAllConnection)) == False:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )

        # # debug
        self.cumul_weigth = self.connection.w.t()
        self.cumul_et = self.target.eligibility_trace.t()
        self.cumul_reward = self.connection.reward_concentration.t()
        self.cumul_pre_post = None
        self.cumul_delta_t = None
        self.cumul_STDP = None

    def update(self, **kwargs) -> None:
        """
        Post-pre learning rule method.
        """
        if self.t < 40 or self.t > 40 :
            BA = 0
        elif self.t == 40:
            print("reward")
            BA = self.BA
            # self.BA = kwargs["reward"]   # amount of biogenic amine released
                                    # NB : argument 'reward' is defined in Network() initialization, not in STDP()

        batch_size = self.source.batch_size

        # Get STDP
        delta_t = self.source.t_spike - self.target.t_spike
        delta_t = delta_t.t()

        tau = torch.where(delta_t > 0, -self.tc_plus, self.tc_minus)
        nu = torch.where(delta_t > 0, self.nu[1], self.nu[0])
        nu = torch.where(delta_t == 0, torch.tensor(0.0), nu)
        STDP = nu * torch.exp(delta_t / tau)

        # Update eligibility trace
        pre_post_spike_occured = self.connection.get_dirac()  # True or 1 if a spike occured either in pre or post neuron, False or 0 otherwise
        update = -self.target.eligibility_trace / self.tc_eligibility_trace + STDP * pre_post_spike_occured
        self.target.eligibility_trace += self.connection.dt * update

        # Update reward
        update = -self.connection.reward_concentration / self.tc_reward
        self.connection.reward_concentration += update * self.connection.dt + self.BA

        # Update weight
        update = self.target.eligibility_trace * self.connection.reward_concentration * self.connection.dt
        self.connection.w = Parameter(torch.max(torch.tensor(self.min_weight), self.connection.w + update))

        # Implement weight decay
        if self.weight_decay:
            self.connection.w -= self.weight_decay * self.connection.w

        self.t += 1
        self.cumul_weigth = torch.cat((self.cumul_weigth, self.connection.w.t()),0)
        self.cumul_et = torch.cat((self.cumul_et,self.target.eligibility_trace.t()),0)
        self.cumul_reward = torch.cat((self.cumul_reward, self.connection.reward_concentration.t()),0)
        if self.cumul_pre_post == None :
            self.cumul_STDP = STDP.t()
            self.cumul_pre_post = pre_post_spike_occured.t()
            self.cumul_delta_t = delta_t.t()
            self.cumul_KC = self.source.t_spike
            self.cumul_EN = self.target.t_spike
        else :
            self.cumul_pre_post = torch.cat((self.cumul_pre_post, pre_post_spike_occured.t()),0)
            self.cumul_STDP = torch.cat((self.cumul_STDP, STDP.t()),0)
            self.cumul_delta_t = torch.cat((self.cumul_delta_t, delta_t.t()),0)
            self.cumul_KC = torch.cat((self.cumul_KC, self.source.t_spike),0)
            self.cumul_EN = torch.cat((self.cumul_EN, self.target.t_spike),0)
