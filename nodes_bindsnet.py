import torch
import sys
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes, CurrentLIFNodes, AdaptiveLIFNodes, IzhikevichNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor

import matplotlib.pyplot as plt 
from bindsnet.analysis.plotting import plot_voltages, plot_spikes

### initialisation 
dt=0.1
simulation_time=500
if len(sys.argv) == 3:
    stimulation = float(sys.argv[2])
else :
    stimulation = 0.1

nodes_network = Network(dt=dt)
input_layer = Input(n=1, traces=True)
nodes_network.add_layer(layer=input_layer, name="Input")
input_monitor = Monitor(obj=input_layer,state_vars=("s"))
nodes_network.add_monitor(monitor=input_monitor,name="input monitor")

### input data
input_data = {"Input" : stimulation * torch.bernoulli(0.1*torch.ones(int(simulation_time/dt), input_layer.n)).byte()}

### LIFNodes
def LIF(nodes_network):
    LIF = LIFNodes(n=2, traces=True)
    nodes_network.add_layer(layer=LIF, name="LIF")
    nodes_network.add_connection(connection=Connection(source=input_layer, target=LIF),source="Input", target="LIF")
    LIF_monitor = Monitor(obj=LIF, state_vars=("s","v"))
    nodes_network.add_monitor(monitor=LIF_monitor, name="LIF monitor")
    return ("LIF", LIF_monitor)

### CurrentLIFNodes
def CurrentLIF(nodes_network):
    CurrentLIF = CurrentLIFNodes(n=1, traces=True)
    nodes_network.add_layer(layer=CurrentLIF, name="CurrentLIF")
    nodes_network.add_connection(connection=Connection(source=input_layer, target=CurrentLIF),source="Input", target="CurrentLIF")
    CurrentLIF_monitor = Monitor(obj=CurrentLIF, state_vars=("s","v"))
    nodes_network.add_monitor(monitor=CurrentLIF_monitor, name="CurrentLIF monitor")
    return ("CurrentLIF", CurrentLIF_monitor)

### AdaptiveLIFNodes
def AdaptiveLIF(nodes_network):
    AdaptiveLIF = AdaptiveLIFNodes(n=1, traces=True)
    nodes_network.add_layer(layer=AdaptiveLIF, name="AdaptiveLIF")
    nodes_network.add_connection(connection=Connection(source=input_layer, target=AdaptiveLIF),source="Input", target="AdaptiveLIF")
    AdaptiveLIF_monitor = Monitor(obj=AdaptiveLIF, state_vars=("s","v"))
    nodes_network.add_monitor(monitor=AdaptiveLIF_monitor, name="AdaptiveLIF monitor")
    return ("AdaptiveLIF", AdaptiveLIF_monitor)

### Izhikevich
def Izhikevich(nodes_network):
    Izhikevich = IzhikevichNodes(n=1, traces=True)
    nodes_network.add_layer(layer=Izhikevich, name="Izhikevich")
    nodes_network.add_connection(connection=Connection(source=input_layer, target=Izhikevich),source="Input", target="Izhikevich")
    Izhikevich_monitor = Monitor(obj=Izhikevich, state_vars=("s","v"))
    nodes_network.add_monitor(monitor=Izhikevich_monitor, name="Izhikevich monitor")
    return ("Izhikevich", Izhikevich_monitor)

### get the nodes type
if len(sys.argv) == 1:
    print("Warning - usage : python nodes_bindsnet.py [nodes type]")
    sys.exit()
elif sys.argv[1] == "LIF":
    (nodes_name, nodes_monitor) = LIF(nodes_network)
elif sys.argv[1] == "CurrentLIF":
    (nodes_name, nodes_monitor) = CurrentLIF(nodes_network)
elif sys.argv[1] == "AdaptiveLIF":
    (nodes_name, nodes_monitor) = AdaptiveLIF(nodes_network)
elif sys.argv[1] == "Izhikevich":
    (nodes_name, nodes_monitor) = Izhikevich(nodes_network)
else : 
    print("Warning - nodes type must be in 'LIF', 'CurrentLIF', 'AdaptiveLIF' or 'Izhikevich'")
    sys.exit()


### run network
nodes_network.run(inputs=input_data, time=simulation_time)
print("[ ",end="")
for i in nodes_monitor.get("v"):
    for j in i:
        print("[["+str(j[0].item())+","+str(j[1].item())+"]],", end=" ")
print(" ]")
plt.ioff()
plot_spikes({"Input": input_monitor.get("s"), nodes_name : nodes_monitor.get("s")})
plot_voltages({nodes_name : nodes_monitor.get("v")}, plot_type="line")
plt.show()