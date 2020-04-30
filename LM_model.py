import torch
import sys
import numpy as np
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection, SparseConnection
from bindsnet.network.monitors import Monitor

import matplotlib.pyplot as plt 
from bindsnet.analysis.plotting import plot_spikes, plot_voltages

### parameters

if len(sys.argv) == 1:
    print("Warning - usage : python LM_model.py [name of image file] [facultative modification to input data]")
    sys.exit()

dt = 0.1
simulation_time = 500 # milliseconds
if len(sys.argv) == 3:
    modification = float(sys.argv[2])
else : 
    modification = 1.0

### get image data

file_image = open(sys.argv[1], "r")
image = []
for l in file_image.readlines():
    l = list(map(lambda x: 100*float(x), l.split()))
    image.append(l)
file_image.close()
image = np.array(image)
image.shape = (1,10,36)
print("Upload image data")

### network initialization based on Ardin et al's article 

landmark_guidance = Network(dt=dt)

PN = Input(n=360, shape=(10,36), traces=True, w=torch.tensor(0.25))
KC = LIFNodes(n=20000, traces=True, w=torch.tensor(2.0))
EN = LIFNodes(n=1, traces=True)
landmark_guidance.add_layer(layer=PN, name="PN")
landmark_guidance.add_layer(layer=KC, name="KC")
landmark_guidance.add_layer(layer=EN, name="EN")

PN_KC = Connection(source=PN, target=KC)
KC_EN = Connection(source=KC, target=EN)
landmark_guidance.add_connection(connection=PN_KC, source="PN", target="KC")
landmark_guidance.add_connection(connection=KC_EN, source="KC", target="EN")

PN_monitor = Monitor(obj=PN, state_vars=("s"), time=simulation_time)
KC_monitor = Monitor(obj=KC, state_vars=("s","v"), time=simulation_time)
EN_monitor = Monitor(obj=EN, state_vars=("s","v"), time=simulation_time)
landmark_guidance.add_monitor(monitor=PN_monitor, name="PN monitor")
landmark_guidance.add_monitor(monitor=KC_monitor, name="KC monitor")
landmark_guidance.add_monitor(monitor=EN_monitor, name="EN monitor")

print("Initialize network")

### run network

print("Run")
input_data = {"PN": torch.from_numpy(modification * np.array([list(map(lambda x: x+np.random.normal(0,0.5), image)) for i in range(int(simulation_time/dt))]))}
landmark_guidance.run(inputs=input_data, time=simulation_time)

spikes = {
    "PN" : PN_monitor.get("s"),
    "KC" : KC_monitor.get("s"),
    "EN" : EN_monitor.get("s")
}
voltages = {
    "KC" : KC_monitor.get("v"),
    "EN" : EN_monitor.get("v")
}

plt.ioff()
plot_spikes(spikes)
plot_voltages(voltages, plot_type="line")
plt.show()