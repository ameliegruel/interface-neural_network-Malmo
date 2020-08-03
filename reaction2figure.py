import math
import torch 
import sys
import numpy as np
from bindsnet.network import Network
from bindsnet.network.nodes import Input, CurrentLIFNodes, LIFNodes, IzhikevichNodes
from bindsnet.network.topology import Connection, SparseConnection, LocalConnection
from bindsnet.network.monitors import Monitor
from ThreeFactorsLearning import STDP, AllToAllConnection, Izhikevich

import matplotlib.pyplot as plt
from bindsnet.analysis.plotting import plot_spikes, plot_voltages, plot_input

### parameters

if len(sys.argv) == 1:
    print("Warning - usage : python LM_model.py [name of learned image file] [name(s) of test image file(s)] [facultative modification to input data]")
    sys.exit()

dt = 1.0
time = 50 # milliseconds
A=1.0
BA=0.5
PN_KC_weight=0.25
min_weight=0.0001
PN_thresh=-40.0
KC_thresh=-25.0
EN_thresh=-40.0
modification=5250.0  # best results with 1.0 for CurrentLIF (for LIF, 0.1) => augmented in order to encode the richess of the input
stimulation_time = 40 # milliseconds 

### get image data

print("Upload image data")

file_image = open(sys.argv[1], "r")
image = []
for l in file_image.readlines():
    l = list(map(lambda x: float(x), l.split()))
    image.append(l)
file_image.close()
image = np.array(image)
image.shape = (1,10,36)

# input_data = {"Input" : torch.from_numpy(image)}
input_data = {"Input": torch.from_numpy(modification * np.array([image if i <= stimulation_time else np.zeros((1,10,36)) for i in range(int(time/dt))]))}

### network initialization based on Ardin et al's article

print("Initialize network")
landmark_guidance = Network(dt=dt)

# layers
input_layer = Input(n=360, shape=(10,36))
PN = Izhikevich(n=360, traces=True, tc_decay=10.0, thresh=PN_thresh, rest=-60.0, C=100, a=0.3, b=-0.2, c=-65, d=8, k=2)
KC = Izhikevich(n=20000, traces=True, tc_decay=10.0, thresh=KC_thresh, rest=-85.0, C=4, a=0.01, b=-0.3, c=-65, d=8, k=0.035)
EN = Izhikevich(n=1, traces=True, tc_decay=10.0, thresh=EN_thresh, rest=-60.0, C=100, a=0.3, b=-0.2, c=-65, d=8, k=2)
landmark_guidance.add_layer(layer=input_layer, name="Input")
landmark_guidance.add_layer(layer=PN, name="PN")
landmark_guidance.add_layer(layer=KC, name="KC")
landmark_guidance.add_layer(layer=EN, name="EN")

# connections
connection_weight = torch.zeros(input_layer.n, PN.n).scatter_(1,torch.tensor([[i,i] for i in range(PN.n)]),1.)
input_PN = Connection(source=input_layer, target=PN, w=connection_weight)

connection_weight = PN_KC_weight * torch.ones(PN.n, KC.n).t()
connection_weight = connection_weight.scatter_(1, torch.tensor([np.random.choice(connection_weight.size(1), size=connection_weight.size(1)-10, replace=False) for i in range(connection_weight.size(0))]).long(), 0.)
PN_KC = AllToAllConnection(source=PN, target=KC, w=connection_weight.t(), tc_synaptic=3.0, phi=0.93)

KC_EN = AllToAllConnection(source=KC, target=EN, w=torch.ones(KC.n, EN.n)*2.0, tc_synaptic=8.0, phi=8.0)
print()
print(KC_EN.w)
print()
landmark_guidance.add_connection(connection=input_PN, source="Input", target="PN")
landmark_guidance.add_connection(connection=PN_KC, source="PN", target="KC")
landmark_guidance.add_connection(connection=KC_EN, source="KC", target="EN")

# learning rule
KC_EN.update_rule = STDP(connection=KC_EN, nu=(-A,-A), tc_eligibility_trace=40.0, tc_plus=15, tc_minus=15, tc_reward=20.0, min_weight=min_weight)

# monitors
input_monitor = Monitor(obj=input_layer, state_vars=("s"))
PN_monitor = Monitor(obj=PN, state_vars=("s","v"))
KC_monitor = Monitor(obj=KC, state_vars=("s","v"))
EN_monitor = Monitor(obj=EN, state_vars=("s","v"))
landmark_guidance.add_monitor(monitor=input_monitor, name="Input monitor")
landmark_guidance.add_monitor(monitor=PN_monitor, name="PN monitor")
landmark_guidance.add_monitor(monitor=KC_monitor, name="KC monitor")
landmark_guidance.add_monitor(monitor=EN_monitor, name="EN monitor")
print("Run")

for phase in [1,2,3]:

    print("Running phase",phase)

    landmark_guidance.reset_state_variables()

    if phase == 2 :
        landmark_guidance.learning = True
    else : 
        landmark_guidance.learning = False

    landmark_guidance.run(inputs=input_data, time=time, reward=BA)

    print()
    print(KC_EN.w)
    print()

    plt.ioff()
    
    spikes = {
        "PN" : PN_monitor.get("s")[-time:],
        "KC" : KC_monitor.get("s")[-time:],
        "EN" : EN_monitor.get("s")[-time:]
    }
    voltages = {
        "PN" : PN_monitor.get("v")[-time:],
        "KC" : KC_monitor.get("v")[-time:],
        "EN" : EN_monitor.get("v")[-time:]
    }

    Pspikes = plot_spikes(spikes)
    for subplot in Pspikes[1]:
        subplot.set_xlim(left=0,right=time)
    Pspikes[1][1].set_ylim(bottom=0, top=KC.n)
    plt.title("Phase "+str(phase)+" - "+sys.argv[1])
    plt.tight_layout()

plt.figure()
plt.plot(range(time+1), torch.tensor(KC_EN.update_rule.cumul_weigth))
plt.title("Evolution of KC_EN weights")

plt.show(block=True)