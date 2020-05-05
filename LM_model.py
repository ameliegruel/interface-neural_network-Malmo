import torch
import sys
import numpy as np
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection, SparseConnection, LocalConnection
from bindsnet.network.monitors import Monitor

import matplotlib.pyplot as plt
from bindsnet.analysis.plotting import plot_spikes, plot_voltages, plot_input

### parameters

if len(sys.argv) == 1:
    print("Warning - usage : python LM_model.py [name of image file] [facultative modification to input data]")
    sys.exit()

dt = 0.1
simulation_time = 50 # milliseconds
if len(sys.argv) == 3:
    modification = float(sys.argv[2])
else :
    modification = 1.0

### get image data

print("Upload image data")

file_image = open(sys.argv[1], "r")
image = []
for l in file_image.readlines():
    l = list(map(lambda x: 100*float(x), l.split()))
    image.append(l)
file_image.close()
image = np.array(image)
image.shape = (10,36)

### network initialization based on Ardin et al's article

print("Initialize network")

landmark_guidance = Network(dt=dt)

input_layer = Input(n=360, shape=(10,36))
PN = LIFNodes(n=360, w=torch.tensor(0.25))
KC = LIFNodes(n=20000, traces=True, w=torch.tensor(2.0))
EN = LIFNodes(n=1, traces=True)
landmark_guidance.add_layer(layer=input_layer, name="Input")
landmark_guidance.add_layer(layer=PN, name="PN")
landmark_guidance.add_layer(layer=KC, name="KC")
landmark_guidance.add_layer(layer=EN, name="EN")

input_PN = LocalConnection(source=input_layer, target=PN, kernel_size=(10,36), stride=(10,36), n_filters=360)
PN_KC = SparseConnection(source=PN, target=KC, sparsity=0.1)
KC_EN = Connection(source=KC, target=EN)
landmark_guidance.add_connection(connection=input_PN, source="Input", target="PN")
landmark_guidance.add_connection(connection=PN_KC, source="PN", target="KC")
landmark_guidance.add_connection(connection=KC_EN, source="KC", target="EN")

input_monitor = Monitor(obj=input_layer, state_vars=("s"))
PN_monitor = Monitor(obj=PN, state_vars=("s","v"), time=simulation_time)
KC_monitor = Monitor(obj=KC, state_vars=("s","v"), time=simulation_time)
EN_monitor = Monitor(obj=EN, state_vars=("s","v"), time=simulation_time)
landmark_guidance.add_monitor(monitor=input_monitor, name="Input monitor")
landmark_guidance.add_monitor(monitor=PN_monitor, name="PN monitor")
landmark_guidance.add_monitor(monitor=KC_monitor, name="KC monitor")
landmark_guidance.add_monitor(monitor=EN_monitor, name="EN monitor")

conn = landmark_guidance.connections[("PN","KC")]
print(torch.mm(conn.w, conn.source.s.unsqueeze(-1).float()).squeeze(-1))

### run network

print("Run")
input_data = {"Input": torch.from_numpy(modification * np.array([list(map(lambda x: x+np.random.normal(0,0.5), image)) for i in range(int(simulation_time/dt))]))}
landmark_guidance.run(inputs=input_data, time=simulation_time)

spikes = {
    "Input" : input_monitor.get("s"),
    "PN" : PN_monitor.get("s"),
    "KC" : KC_monitor.get("s"),
    "EN" : EN_monitor.get("s")
}
voltages = {
    "PN" : PN_monitor.get("v"),
    "KC" : KC_monitor.get("v"),
    "EN" : EN_monitor.get("v")
}

# print(len(spikes["PN"]))

plt.ioff()
Pspikes = plot_spikes(spikes)
for subplot in Pspikes[1]:
    subplot.set_xlim(left=0,right=simulation_time)
Pspikes[1][2].set_ylim(bottom=0, top=KC.n)

Vspikes = plot_voltages(voltages, plot_type="line")
for v_subplot in Vspikes[1]:
    v_subplot.set_xlim(left=0, right=simulation_time)

plot_input(image=torch.from_numpy(image[0]), inpt=spikes["Input"][0][0])

plt.show()