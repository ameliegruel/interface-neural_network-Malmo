from abc import ABC
from typing import Union, Optional, Sequence

import torch
from torch.nn import Module, Parameter
import numpy as np
from bindsnet.learning.learning import NoOp
from bindsnet.network.nodes import Nodes
from bindsnet.network.topology import (
    AbstractConnection,
    Connection,
    Conv2dConnection,
    LocalConnection,
)


class AllToAllConnection(ABC, Module):
    
    def __init__(
        self,
        source: Nodes,
        target: Nodes,
        w: None,
        tc_synaptic: float = 0.0,
        phi: float = 0.0,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        """
        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param nu: Learning rate for both pre- and post-synaptic events.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
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
        self.reduction = reduction

        self.update_rule = kwargs.get("update_rule", NoOp)
        self.wmin = kwargs.get("wmin", -np.inf)
        self.wmax = kwargs.get("wmax", np.inf)
        self.norm = kwargs.get("norm", None)
        self.decay = kwargs.get("decay", None)

        # Learning rule
        if self.update_rule is None:
            self.update_rule = NoOp

        self.update_rule = self.update_rule(
            connection=self,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs
        )

        # Weights
        if w is None:
            if self.wmin == -np.inf or self.wmax == np.inf:
                w = torch.clamp(torch.rand(source.n, target.n), self.wmin, self.wmax)
            else:
                w = self.wmin + torch.rand(source.n, target.n) * (self.wmax - self.wmin)

        self.w = Parameter(w, requires_grad=False)
        self.b = Parameter(kwargs.get("b", torch.zeros(target.n)), requires_grad=False)

        # Parameters used to update synaptic input
        self.active_neurotransmitters = torch.zeros(self.source.n, self.target.n)
        self.tc_synaptic = tc_synaptic
        self.phi = phi
        self.v_rev = 0

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
        # pre_spike_occured = s.float() # size [1,360]
        # print(pre_spike_occured.shape)
        pre_post_spike_occured = self.get_dirac()
        update = - self.active_neurotransmitters / self.tc_synaptic + self.phi * pre_post_spike_occured
        self.active_neurotransmitters += update
        # print(self.active_neurotransmitters.shape)

        # Get input 
        # print(self.source.v.shape)
        # # output = self.active_neurotransmitters @ self.w
        # print(output.shape) # size [360,20000]
        output = (self.v_rev - self.source.v) @ (self.w * self.active_neurotransmitters)
        print(output)
        return output

    def update(self, **kwargs) -> None:
        # language=rst
        """
        Compute connection's update rule.

        Keyword arguments:

        :param bool learning: Whether to allow connection updates.
        :param ByteTensor mask: Boolean mask determining which weights to clamp to zero.
        """
        learning = kwargs.get("learning", True)

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
    

######################################################""


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
        tc_minus: float = 0.0,
        tc_plus: float = 0.0,
        threshold: float = 0.0,
        **kwargs
    ) -> None:
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
        self.threshold = threshold

        # Initialize eligibility trace, time constants and reward
        if not hasattr(self.target, "eligibility_trace"):
            self.target.eligibility_trace = torch.zeros(*self.connection.w.shape) 
        self.tc_eligibility_trace = tc_eligibility_trace
        self.tc_minus = torch.tensor(tc_minus)
        self.tc_plus = torch.tensor(tc_plus)
        if not hasattr(self.connection, "reward_concentration"):
            self.connection.reward_concentration = torch.zeros(*self.connection.w.shape) # initialize the extracellular concentration of biogenic amine
        self.tc_reward = tc_reward
        self.reward = 0 # nul for every t except t=40 ms

        # Parameter update reduction across minibatch dimension.
        if reduction is None:
            reduction = torch.mean
        self.reduction = reduction

        # Weight decay.
        self.weight_decay = weight_decay

        assert (
            self.source.traces and self.target.traces
        ), "Both pre- and post-synaptic nodes must record spike traces."

        if isinstance(connection, (Connection, LocalConnection, Conv2dConnection, AllToAllConnection)) == False:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )

        # debug
        self.cumul_weigth = self.connection.w.t()
        self.cumul_et = self.target.eligibility_trace.t()
        self.cumul_reward = self.connection.reward_concentration.t()

    def update(self, **kwargs) -> None:
        """
        Post-pre learning rule method.
        """
        if self.t == 40:
            print("reward")
            self.reward = kwargs["reward"]   # amount of biogenic amine released 
                                    # NB : argument 'reward' is defined in Network() initialization, not in STDP()
        elif self.t > 40:
            self.reward = 0

        batch_size = self.source.batch_size

        pre_x = self.source.x.view(-1).unsqueeze(1)
        post_x = self.target.x.view(-1).unsqueeze(1)

        # Get STDP
        delta_t = pre_x - post_x
        delta_t = torch.where(delta_t == 0, torch.tensor(-50.0), delta_t) # if delta_t == 0, STDP == 0 (interruption in the function)
        tau = torch.where(delta_t > 0, -self.tc_plus, self.tc_minus)
        nu = torch.where(delta_t > 0, self.nu[1], self.nu[0])
        STDP = nu * torch.exp(delta_t / tau)
        
        # Update eligibility trace
        pre_post_spike_occured = self.connection.get_dirac()  # True or 1 if a spike occured either in pre or post neuron, False or 0 otherwise
        update = -self.target.eligibility_trace / self.tc_eligibility_trace + STDP * pre_post_spike_occured
        self.target.eligibility_trace += self.connection.dt * update

        # Update reward
        update = -self.connection.reward_concentration / self.tc_reward
        self.connection.reward_concentration += update * self.connection.dt + self.reward

        # Update weight
        update = self.target.eligibility_trace * self.connection.reward_concentration * self.connection.dt
        self.connection.w = Parameter(torch.max(torch.tensor(self.threshold), self.connection.w + update))

        # Implement weight decay
        if self.weight_decay:
            self.connection.w -= self.weight_decay * self.connection.w

        self.t += 1
        self.cumul_weigth = torch.cat((self.cumul_weigth, self.connection.w.t()),0)
        self.cumul_et = torch.cat((self.cumul_et,self.target.eligibility_trace.t()),0)
        self.cumul_reward = torch.cat((self.cumul_reward, self.connection.reward_concentration.t()),0)
        # print(self.connection.w)