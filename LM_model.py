import datetime
import torch 
import sys
import numpy as np
from bindsnet.network import Network
from bindsnet.network.nodes import Input, CurrentLIFNodes, LIFNodes
from bindsnet.network.topology import Connection, SparseConnection, LocalConnection
from bindsnet.network.monitors import Monitor
from ThreeFactorsLearning import STDP, AllToAllConnection

import matplotlib.pyplot as plt
from bindsnet.analysis.plotting import plot_spikes, plot_voltages, plot_input

### parameters

if len(sys.argv) == 1:
    print("Warning - usage : python LM_model.py [name of learned image file] [name(s) of test image file(s)]")
    sys.exit()

dt = 1.0
learning_time = 50 # milliseconds
test_time = 50 # milliseconds
modification = 1.0 # best results with this modification for CurrentLIF (for LIF, 0.1)
try:
    A = int(sys.argv[-1])*0.1
    threshold = int(sys.argv[-1])*0.0001
    last_file_index = len(sys.argv) - 2
except ValueError:
    A = 1.0
    threshold = 0.0001
    last_file_index = len(sys.argv)


### get image data

print("Upload image data")

input_data = {"Learning": None, "Test": {}}
list_files = sys.argv[1:last_file_index]
for i in range(len(list_files)):
    file_image = open(list_files[i], "r")
    image = []
    for l in file_image.readlines():
        l = list(map(lambda x: 10*float(x), l.split()))
        image.append(l)
    file_image.close()
    image = np.array(image)
    image.shape = (1,10,36)

    if i == 0:
        print(list_files[i],"=> learning")
        input_data["Learning"] = {"Input": torch.from_numpy(modification * np.array([image for i in range(int(learning_time/dt))]))}
    else : 
        print(list_files[i], "=> test")
        input_data["Test"][list_files[i]] = {"Input": torch.from_numpy(modification * np.array([image for i in range(int(test_time/dt))]))}


### network initialization based on Ardin et al's article

print("Initialize network")
begin_time = datetime.datetime.now()

landmark_guidance = Network(dt=dt)

# layers
input_layer = Input(n=360, shape=(10,36))
PN = CurrentLIFNodes(n=360, traces=True, tc_decay=10.0)
KC = CurrentLIFNodes(n=20000, traces=True, tc_decay=10.0)
EN = CurrentLIFNodes(n=1, traces=True, tc_decay=10.0)
landmark_guidance.add_layer(layer=input_layer, name="Input")
landmark_guidance.add_layer(layer=PN, name="PN")
landmark_guidance.add_layer(layer=KC, name="KC")
landmark_guidance.add_layer(layer=EN, name="EN")

# connections
connection_weight = torch.zeros(input_layer.n, PN.n).scatter_(1,torch.tensor([[i,i] for i in range(PN.n)]),1.)
input_PN = Connection(source=input_layer, target=PN, w=connection_weight)

connection_weight = 0.25*torch.ones(PN.n, KC.n).t()
connection_weight = connection_weight.scatter_(1, torch.tensor([np.random.choice(connection_weight.size(1), size=connection_weight.size(1)-10, replace=False) for i in range(connection_weight.size(0))]).long(), 0.)
PN_KC = AllToAllConnection(source=PN, target=KC, w=connection_weight.t(), tc_synaptic=3.0, phi=0.93)

KC_EN = AllToAllConnection(source=KC, target=EN, w=torch.ones(KC.n, EN.n)*2.0, tc_synaptic=8.0, phi=8.0)
landmark_guidance.add_connection(connection=input_PN, source="Input", target="PN")
landmark_guidance.add_connection(connection=PN_KC, source="PN", target="KC")
landmark_guidance.add_connection(connection=KC_EN, source="KC", target="EN")

# learning rule
KC_EN.update_rule = STDP(connection=KC_EN, nu=(-A,-A), tc_eligibility_trace=40.0, tc_plus=15, tc_minus=15, tc_reward=20.0, threshold=threshold)

# monitors
input_monitor = Monitor(obj=input_layer, state_vars=("s"))
PN_monitor = Monitor(obj=PN, state_vars=("s","v"))
KC_monitor = Monitor(obj=KC, state_vars=("s","v"))
EN_monitor = Monitor(obj=EN, state_vars=("s","v"))
landmark_guidance.add_monitor(monitor=input_monitor, name="Input monitor")
landmark_guidance.add_monitor(monitor=PN_monitor, name="PN monitor")
landmark_guidance.add_monitor(monitor=KC_monitor, name="KC monitor")
landmark_guidance.add_monitor(monitor=EN_monitor, name="EN monitor")
print(datetime.datetime.now()-begin_time)

### run network : learning of 1 view
begin_time = datetime.datetime.now()

print("Run - learning view")

landmark_guidance.learning = True
landmark_guidance.run(inputs=input_data["Learning"], time=learning_time, reward=0.5)
landmark_guidance.learning = False

print("> View learned")

plt.figure()
plt.plot(range(learning_time+1), torch.tensor(KC_EN.update_rule.cumul_weigth))
plt.title("Evolution of KC_EN weights for A="+str(A)+" and thresh="+str(threshold))
# plt.savefig("./manual_tuning/weights_nu"+str(A)+"_thresh"+str(threshold)+".png")

plt.figure()
plt.plot(range(learning_time+1), torch.tensor(KC_EN.update_rule.cumul_et))
plt.title("Evolution of KC_EN eligibility traces for A="+str(A)+" and thresh="+str(threshold))
# plt.savefig("./manual_tuning/eligibility_nu"+str(A)+"_thresh"+str(threshold)+".png")

# plt.figure()
# plt.plot(range(learning_time+1), torch.tensor(PN_KC.cumul_weight))
# plt.title("Evolution of PN_KC weights")

plt.figure()
plt.plot(range(learning_time+1), torch.tensor(KC_EN.update_rule.cumul_reward))
plt.title("Evolution of KC_EN reward concentrations")

### run network : test on one or more views

print("Run - test of one or more views")
view = {"name": None, "mean_EN" : None}

plt.ioff()
for (name, data) in input_data["Test"].items():
    landmark_guidance.reset_state_variables()
    landmark_guidance.run(inputs=data, time=test_time)

    spikes = {
        "Input" : input_monitor.get("s")[-test_time:],
        "PN" : PN_monitor.get("s")[-test_time:],
        "KC" : KC_monitor.get("s")[-test_time:],
        "EN" : EN_monitor.get("s")[-test_time:]
    }
    voltages = {
        "PN" : PN_monitor.get("v")[-test_time:],
        "KC" : KC_monitor.get("v")[-test_time:],
        "EN" : EN_monitor.get("v")[-test_time:]
    }

    print(name, ":  nb spikes =", len(torch.nonzero(spikes["EN"])))

    if view["mean_EN"] == None or len(torch.nonzero(spikes["EN"])) < view["mean_EN"] : 
        view["mean_EN"] = len(torch.nonzero(spikes["EN"]))
        view["name"] = name

    Pspikes = plot_spikes(spikes)
    for subplot in Pspikes[1]:
        subplot.set_xlim(left=0,right=test_time)
    Pspikes[1][2].set_ylim(bottom=0, top=KC.n)
    plt.suptitle("Results for " + name)

    Pvoltages = plot_voltages(voltages, plot_type="line")
    for v_subplot in Pvoltages[1]:
        v_subplot.set_xlim(left=0, right=test_time)
    Pvoltages[1][2].set_ylim(bottom=min(-70, min(voltages["EN"])), top=max(-50, max(voltages["EN"])))
    plt.suptitle("Results for " + name)

    plt.show(block=False)

print("Most familiar view:", view["name"])

plt.show(block=True)