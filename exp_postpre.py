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
    print("Warning - usage : python LM_model.py [name of learned image file] [name(s) of test image file(s)] [facultative modification to input data]")
    sys.exit()

dt = 1.0
learning_time = 50 # milliseconds
test_time = 50 # milliseconds
try:
    modification = float(sys.argv[-1])
    last_file_index = len(sys.argv)-1
except ValueError:
    modification = 1.0
    last_file_index = len(sys.argv)

### get image data

print("Upload image data")

file_image = open(sys.argv[1], "r")
image = []
for l in file_image.readlines():
    l = list(map(lambda x: 10*float(x), l.split()))
    image.append(l)
file_image.close()
image = np.array(image)
image.shape = (1,10,36)

input_data_learning = {"Input" : torch.from_numpy(modification * np.array([image for i in range(int(learning_time/dt))]))}
input_data_test = {"Input": torch.from_numpy(modification * np.array([image for i in range(int(test_time/dt))]))}


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
input_PN = LocalConnection(source=input_layer, target=PN, kernel_size=(10,36), stride=(10,36), n_filters=360)

connection_weight = 0.25*torch.ones(PN.n, KC.n).t()
connection_weight = connection_weight.scatter_(1, torch.tensor([np.random.choice(connection_weight.size(1), size=connection_weight.size(1)-10, replace=False) for i in range(connection_weight.size(0))]).long(), 0.).t()
indices = torch.nonzero(connection_weight).t()
# connection_weight = torch.sparse.FloatTensor(indices, connection_weight[indices[0],indices[1]], connection_weight.size())
PN_KC = AllToAllConnection(source=PN, target=KC, w=connection_weight, tc_synaptic=3.0, phi=0.93)

KC_EN = AllToAllConnection(source=KC, target=EN, w=torch.ones(KC.n, EN.n)*2.0, tc_synaptic=8.0, phi=8.0)
landmark_guidance.add_connection(connection=input_PN, source="Input", target="PN")
landmark_guidance.add_connection(connection=PN_KC, source="PN", target="KC")
landmark_guidance.add_connection(connection=KC_EN, source="KC", target="EN")

# learning rule
KC_EN.update_rule = STDP(connection=KC_EN, nu=(-0.2,-0.2), tc_eligibility_trace=40.0, tc_plus=15, tc_minus=15, tc_reward=20.0)

# monitors
input_monitor = Monitor(obj=input_layer, state_vars=("s"))
PN_monitor = Monitor(obj=PN, state_vars=("s","v"))
KC_monitor = Monitor(obj=KC, state_vars=("s","v"))
EN_monitor = Monitor(obj=EN, state_vars=("s","v"))
landmark_guidance.add_monitor(monitor=input_monitor, name="Input monitor")
landmark_guidance.add_monitor(monitor=PN_monitor, name="PN monitor")
landmark_guidance.add_monitor(monitor=KC_monitor, name="KC monitor")
landmark_guidance.add_monitor(monitor=EN_monitor, name="EN monitor")
print("Initialisation:", datetime.datetime.now()-begin_time)


print("Run")

landmark_guidance.learning = False
landmark_guidance.run(inputs=input_data_test, time=test_time)

plt.ioff()

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

# mean = 0
# for i in spikes['KC']:
#     count = 0
#     for j in i.view(-1) :
#         if j.item() == True:
#             count += 1
#     print(count)
#     mean += count
# mean = mean / len(spikes["KC"])
# print("mean:",mean)

print("Test :  mean EN =", torch.mean(voltages["EN"]), " -  max EN =", torch.max(voltages["EN"]))
print("nb spikes =", len(torch.nonzero(spikes["EN"])))

Pspikes = plot_spikes(spikes)
for subplot in Pspikes[1]:
    subplot.set_xlim(left=0,right=test_time)
Pspikes[1][2].set_ylim(bottom=0, top=KC.n)
plt.suptitle("Results before learning")
# plt.savefig("./result_random_nav/1s_spikes_"+sys.argv[1]+".png")

Pvoltages = plot_voltages(voltages, plot_type="line")
for v_subplot in Pvoltages[1]:
    v_subplot.set_xlim(left=0, right=test_time)
Pvoltages[1][2].set_ylim(bottom=min(-70, min(voltages["EN"])), top=-50)
plt.suptitle("Results before learning")
# plt.savefig("./result_random_nav/1s_voltages_"+sys.argv[1]+".png")
## run network : learning of 1 view
# begin_time = datetime.datetime.now()

# landmark_guidance.learning = True
# landmark_guidance.run(inputs=input_data_learning, time=learning_time, reward=0.5)
# w_after = KC_EN.w

# # for i in range(len(KC_EN.w)):
# #     if w_before[i].item() != w_after[i].item():
# #         print("> connection", i, ": ", w_before[i].item(), "puis", w_after[i].item())

# print("> View learned")

# spikes = {
#     "Input" : input_monitor.get("s"),
#     "PN" : PN_monitor.get("s"),
#     "KC" : KC_monitor.get("s"),
#     "EN" : EN_monitor.get("s")
# }
# voltages = {
#     "PN" : PN_monitor.get("v"),
#     "KC" : KC_monitor.get("v"),
#     "EN" : EN_monitor.get("v")
# }

# print("Learning :   ", "mean EN =", torch.mean(voltages["EN"]), " -  min EN =", torch.min(voltages["EN"]))
# print("nb spikes =", len(torch.nonzero(spikes["EN"])))

# Pspikes = plot_spikes(spikes)
# for subplot in Pspikes[1]:
#     subplot.set_xlim(left=0,right=learning_time)
# Pspikes[1][2].set_ylim(bottom=0, top=KC.n)
# plt.suptitle("Learning")
# # plt.savefig("./result_random_nav/1s_spikes_"+sys.argv[1]+".png")

# Pvoltages = plot_voltages(voltages, plot_type="line")
# for v_subplot in Pvoltages[1]:
#     v_subplot.set_xlim(left=0, right=learning_time)
# Pvoltages[1][2].set_ylim(bottom=min(-70, torch.min(voltages["EN"])), top=max(-50, torch.max(voltages["EN"])))
# plt.suptitle("Learning")
# plt.savefig("./result_random_nav/1s_voltages_"+sys.argv[1]+".png")

# plt.show(block=False)

plt.show(block=True)