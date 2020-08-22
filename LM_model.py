import datetime
import torch 
import sys
import numpy as np
from bindsnet.network import Network
from bindsnet.network.nodes import Input, CurrentLIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from AntLearning import STDP, AllToAllConnection, Izhikevich

import matplotlib.pyplot as plt
from bindsnet.analysis.plotting import plot_spikes, plot_voltages, plot_input

def LM_model(
    plot_parameters=False, 
    plot_results=False, 
    arguments=False, 
    info_PN = False,
    figures=None,
    A=1.0,
    BA=0.5,
    PN_KC_weight=0.25,
    min_weight=0.0001,
    PN_thresh=-40.0,
    KC_thresh=-25.0,
    EN_thresh=-40.0,
    modification=5.0,  # best results with 1.0 for CurrentLIF (for LIF, 0.1) => augmented in order to encode the richess of the input
    stimulation_time = 40 # milliseconds 
    ) :

    begin_time = datetime.datetime.now()

    ### parameters

    dt = 1.0
    learning_time = 50 # milliseconds
    test_time = 50 # milliseconds
    
    if arguments == True :
        if len(sys.argv) == 1:
            print("Warning - usage : python LM_model.py [name of learned image file] [name(s) of test image file(s)]")
            sys.exit()

        try:
            A = int(sys.argv[-1])*0.1
            min_weight = int(sys.argv[-1])*0.0001
            last_file_index = len(sys.argv) - 2
        except ValueError:
            last_file_index = len(sys.argv)
        list_files = sys.argv[1:last_file_index]
    
    else :
        list_files = figures


    ### get image data
    
    print("Upload image data")

    input_data = {"Learning": None, "Test": {}}
    for i in range(len(list_files)):
        file_image = open(list_files[i], "r")
        image = []
        for l in file_image.readlines():
            l = list(map(lambda x: float(x), l.split()))
            image.append(l)
        file_image.close()
        image = np.array(image)
        image.shape = (1,10,36)

        if i == 0:
            print(list_files[i],"=> learning")
            input_data["Learning"] = {"Input": torch.from_numpy(modification * np.array([image if i <= stimulation_time else np.zeros((1,10,36)) for i in range(int(learning_time/dt))]))}
        else : 
            print(list_files[i], "=> test")
            input_data["Test"][list_files[i]] = {"Input": torch.from_numpy(modification * np.array([image if i <= stimulation_time else np.zeros((1,10,36)) for i in range(int(test_time/dt))]))}


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
    print(datetime.datetime.now()-begin_time)

    ### run network : learning of 1 view
    begin_time = datetime.datetime.now()

    print("Run - learning view")

    landmark_guidance.learning = True
    landmark_guidance.run(inputs=input_data["Learning"], time=learning_time, reward=BA, n_timesteps=test_time/dt)
    landmark_guidance.learning = False

    print()
    print(KC_EN.w)
    print()

    print("> View learned")

    if plot_parameters == True :
        plt.figure()
        plt.plot(range(learning_time+1), torch.tensor(KC_EN.update_rule.cumul_weigth))
        plt.title("Evolution of KC_EN weights for A="+str(A)+" and thresh="+str(min_weight))
        # plt.savefig("./manual_tuning/weights_nu"+str(A)+"_thresh"+str(min_weight)+".png")

        plt.figure()
        plt.plot(range(learning_time+1), torch.tensor(KC_EN.update_rule.cumul_et))
        plt.title("Evolution of KC_EN eligibility traces for A="+str(A)+" and thresh="+str(min_weight))
        # plt.savefig("./manual_tuning/eligibility_nu"+str(A)+"_thresh"+str(min_weight)+".png")

        plt.figure()
        plt.plot(range(learning_time), torch.tensor(KC_EN.update_rule.cumul_delta_t), "b", range(learning_time), torch.tensor(KC_EN.update_rule.cumul_KC), "r", range(learning_time), torch.tensor(KC_EN.update_rule.cumul_EN), "g")
        # plt.plot(range(learning_time), torch.tensor(KC_EN.update_rule.cumul_delta_t))
        plt.title("Evolution of delta_t")

        plt.figure()
        plt.plot(range(learning_time), torch.tensor(KC_EN.update_rule.cumul_STDP))
        plt.title("Evolution of STDP")

        plt.figure()
        plt.plot(range(learning_time), torch.tensor(KC_EN.update_rule.cumul_pre_post))
        plt.title("Evolution of pre_post_spikes")

        plt.figure()
        plt.plot(range(learning_time), torch.tensor(PN_KC.cumul_I))
        plt.title("Evolution of I KC")

        plt.figure()
        plt.plot(range(learning_time), torch.tensor(KC_EN.cumul_I))
        plt.xlim(left=0, right=learning_time)
        plt.title("Evolution of I EN")

        plt.show(block=False)

    ### run network : test on one or more views

    print("Run - test of one or more views")
    view = {"name": None, "mean_EN" : None}
    nb_spikes = []

    plt.ioff()
    for (name, data) in input_data["Test"].items():
        landmark_guidance.reset_state_variables()
        landmark_guidance.run(inputs=data, time=test_time, n_timesteps=test_time/dt)

        spikes = {
            "PN" : PN_monitor.get("s")[-test_time:],
            "KC" : KC_monitor.get("s")[-test_time:],
            "EN" : EN_monitor.get("s")[-test_time:]
        }
        voltages = {
            "PN" : PN_monitor.get("v")[-test_time:],
            "KC" : KC_monitor.get("v")[-test_time:],
            "EN" : EN_monitor.get("v")[-test_time:]
        }

        if info_PN == True:
            frequences = []
            for nodes in spikes["PN"].squeeze().t():
                frequences.append(len(torch.nonzero(nodes)))
            frequences = torch.tensor(frequences).float()
            print("Mean spikes PN :", torch.mean(frequences), "- Max :", torch.max(frequences), "- Min :", torch.min(frequences))

        print(name, ":  nb spikes EN =", len(torch.nonzero(spikes["EN"])))
        nb_spikes.append(len(torch.nonzero(spikes["EN"]))) 

        if view["mean_EN"] == None or len(torch.nonzero(spikes["EN"])) < view["mean_EN"] : 
            view["mean_EN"] = len(torch.nonzero(spikes["EN"]))
            view["name"] = name

        if plot_results == True : 
            Pspikes = plot_spikes(spikes)
            for subplot in Pspikes[1]:
                subplot.set_xlim(left=0,right=test_time)
            Pspikes[1][1].set_ylim(bottom=0, top=KC.n)
            plt.suptitle("Results for " + name)

            # Pvoltages = plot_voltages(voltages, plot_type="line")
            # for v_subplot in Pvoltages[1]:
            #     v_subplot.set_xlim(left=0, right=test_time)
            # Pvoltages[1][2].set_ylim(bottom=min(-70, min(voltages["EN"])), top=max(-50, max(voltages["EN"])))
            # plt.suptitle("Results for " + name)

            plt.show(block=False)

    print("Most familiar view:", view["name"])

    plt.show(block=True)
    print(datetime.datetime.now() - begin_time)

    if nb_spikes[0] == nb_spikes[1] == nb_spikes[2]:
        return (view['name'], True)
    else :
        return (view["name"], False)



LM_model(plot_parameters=True, plot_results=True, arguments=True, KC_thresh=-25, A=1.0, PN_KC_weight=0.25, modification=5250.0, min_weight=0.0001, BA=0.5)