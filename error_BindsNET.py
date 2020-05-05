import torch
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection, SparseConnection
from bindsnet.network.monitors import Monitor

# parameters
simulation_time = 50
dt=0.1



### Example of error when the target of a connection has a specified shape
net = Network(dt=dt)

source_layer = Input(n=360, shape=(10,36))
target_layer = LIFNodes(n=360, shape=(10,36))
connection_source_target = Connection(source=source_layer, target=target_layer)
source_monitor = Monitor(obj=source_layer, state_vars=("s"))
target_monitor = Monitor(obj=target_layer, state_vars=("s","v"))

net.add_layer(layer=source_layer, name="source")
net.add_layer(layer=target_layer, name="target")
net.add_connection(connection=connection_source_target, source="source", target="target")
net.add_monitor(monitor=source_monitor, name="source monitor")
net.add_monitor(monitor=target_monitor, name="target monitor")

input_data = {"source" : torch.bernoulli(0.1*torch.ones(int(simulation_time/dt), source_layer.shape[0], source_layer.shape[1])).byte()}
print(input_data["source"].size())  # torch.Size([500, 10, 36])

net.run(inputs=input_data, time=simulation_time)

"""
Error : 

Traceback (most recent call last):
  File "error_BindsNET.py", line 31, in <module>
    net.run(inputs=input_data, time=simulation_time)
  File "C:\Program Files\Python367\lib\site-packages\bindsnet\network\network.py", line 390, in run
    current_inputs.update(self._get_inputs())
  File "C:\Program Files\Python367\lib\site-packages\bindsnet\network\network.py", line 234, in _get_inputs
    inputs[c[1]] += self.connections[c].compute(source.s)
  File "C:\Program Files\Python367\lib\site-packages\bindsnet\network\topology.py", line 178, in compute
    post = s.float().view(s.size(0), -1) @ self.w + self.b
RuntimeError: size mismatch, m1: [10 x 36], m2: [360 x 360] at C:\w\1\s\windows\pytorch\aten\src\TH/generic/THTensorMath.cpp:136
"""




### Example of error when implementing SparseConnection
net = Network(dt=dt)

source_layer = Input(n=360, shape=(10,36))
target_layer = LIFNodes(n=360)
connection_source_target = SparseConnection(source=source_layer, target=target_layer, sparsity=0.1)
source_monitor = Monitor(obj=source_layer, state_vars=("s"))
target_monitor = Monitor(obj=target_layer, state_vars=("s","v"))

net.add_layer(layer=source_layer, name="source")
net.add_layer(layer=target_layer, name="target")
net.add_connection(connection=connection_source_target, source="source", target="target")
net.add_monitor(monitor=source_monitor, name="source monitor")
net.add_monitor(monitor=target_monitor, name="target monitor")

input_data = {"source" : torch.bernoulli(0.1*torch.ones(int(simulation_time/dt), source_layer.shape[0], source_layer.shape[1])).byte()}
print(input_data["source"].size())  # torch.Size([500, 10, 36])

net.run(inputs=input_data, time=simulation_time)

"""
Error : 

Traceback (most recent call last):
  File "error_BindsNET.py", line 63, in <module>
    net.run(inputs=input_data, time=simulation_time)
  File "C:\Program Files\Python367\lib\site-packages\bindsnet\network\network.py", line 343, in run
    current_inputs.update(self._get_inputs())
  File "C:\Program Files\Python367\lib\site-packages\bindsnet\network\network.py", line 234, in _get_inputs
    inputs[c[1]] += self.connections[c].compute(source.s)
  File "C:\Program Files\Python367\lib\site-packages\bindsnet\network\topology.py", line 785, in compute
    return torch.mm(self.w, s.unsqueeze(-1).float()).squeeze(-1)
RuntimeError: addmm: matrices expected, got 3D tensor
"""