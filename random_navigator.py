from __future__ import print_function
from __future__ import division

from builtins import range
from past.utils import old_div
import MalmoPython
import os
import sys
import time
import random
import json
from math import sqrt, cos, sin, tan, atan2, degrees, radians
from skimage import exposure, transform
import numpy as np
from PIL import Image
import datetime

import torch
from bindsnet.network import Network
from bindsnet.network.nodes import CurrentLIFNodes, Input, LIFNodes
from bindsnet.network.topology import Connection, LocalConnection
from bindsnet.network.monitors import Monitor
import matplotlib.pyplot as plt 
from bindsnet.analysis.plotting import plot_voltages, plot_spikes

if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)




#####################################################################################
####################### WORLD INITIALISATION ########################################
#####################################################################################


ArenaSide = 1000 # the agent's size is 1 block = 1 cm (for an ant) => the agent evolves in an arena of diameter 10m*10m
ArenaFloor = 2
xPos = 0
zPos = 0

video_height = 600
video_width = 800

# Create random objects in the arena

def GenRandomObject(xPos, zPos, ArenaSide, ArenaFloor):
    print("Waiting for the world to load ", end=' ')
    # we want 2 obstacles in a 10cm*10cm region => for 1cm*1cm = 1block*1block, we need 0.05obstacles
    nb = int(0.01*ArenaSide*ArenaSide)
    RandomObjectXML = ""
    BlockType = "redstone_block"
    ObjHeight = [2,3,4,5]
    ObjWidth = [1,2]
    max = old_div(ArenaSide,2)
    PreviousCoord = [[x,z] for x in range(xPos-1, xPos+2) for z in range(zPos-1, zPos+2)]
    for i in range(nb) :
        if i % 50 == 0 :
            print(".", end="")
        height = random.choice(ObjHeight)
        width = random.choice(ObjWidth)
        [x0,z0] = [xPos, zPos]
        while [x0,z0] in PreviousCoord :
            x0 = random.randint(-max,max)
            z0 = random.randint(-max,max)
        for [x,y,z] in [[x0+x, ArenaFloor+y, z0+z] for x in range(width) for y in range(height) for z in range(width)]:
            RandomObjectXML += '<DrawBlock type="' + BlockType + '" x="' + str(x) + '" y="' + str(y) + '" z="' + str(z) + '"/>'
            PreviousCoord.append([x,z])
    print()
    return RandomObjectXML






#########################################################################################################
############################################# XML #######################################################
#########################################################################################################

missionXML='''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
            
              <About>
                <Summary>Interface</Summary>
              </About>
              
            <ServerSection>
              <ServerInitialConditions>
                <Time>
                    <StartTime>1000</StartTime>
                    <AllowPassageOfTime>false</AllowPassageOfTime>
                </Time>
                <Weather>clear</Weather>
              </ServerInitialConditions>
              <ServerHandlers>
                    <FlatWorldGenerator generatorString="3;1,12*35:5;2;"/>
                    <DrawingDecorator>
                        <DrawCuboid type="air" x1="''' + str(old_div(-ArenaSide,2)) + '''" y1="''' + str(ArenaFloor) + '''" z1="''' + str(old_div(-ArenaSide,2)) + '''" x2="''' + str(old_div(ArenaSide,2)) + '''" y2="20" z2="''' + str(old_div(ArenaSide,2)) + '''"/>
                        ''' + GenRandomObject(xPos, zPos, ArenaSide, ArenaFloor) + '''
                    </DrawingDecorator>
                    <ServerQuitFromTimeUp timeLimitMs="50000"/>
                    <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>
              
              <AgentSection mode="Survival">
                <Name>MalmoTutorialBot</Name>
                <AgentStart>
                    <Placement x="0" y="''' + str(ArenaFloor) + '''" z="0" yaw="0"/>
                    <Inventory>
                        <InventoryItem slot="8" type="diamond_pickaxe"/>
                    </Inventory>
                </AgentStart>
                <AgentHandlers>
                    <ChatCommands/>
                    <ObservationFromFullStats/>
                    <ObservationFromGrid>
                        <Grid name="FrontEnv21x21">
                            <min x="-10" y="0" z="-10"/>
                            <max x="10" y="0" z="10"/>
                        </Grid>
                    </ObservationFromGrid>
                    <ContinuousMovementCommands turnSpeedDegs="180"/>
                    <InventoryCommands/>
                    <VideoProducer want_depth="true">
                        <Width>'''+str(video_width)+'''</Width>
                        <Height>'''+str(video_height)+'''</Height>
                    </VideoProducer>
                </AgentHandlers>
              </AgentSection>
            </Mission>'''







#############################################################################################
############################ GET SENSORY INFORMATION ########################################
#############################################################################################


# Get the correct Yaw (strictly between 180 and -180) at each time

def getYaw(yaw) :
    if yaw / 180 > 1 :
        yaw = yaw - 360
    elif yaw / 180 < -1 :
        yaw = yaw + 360
    return yaw


# Get the coordinates corresponding to the visual field

def getA(lenght, angle, pos) :
    return lenght * cos(radians(angle)) + pos

def getB(a, angle, pos):
    return (a-pos) * tan(radians(angle)) + pos


def GetVisualField(side,yaw,vf=70) :
    # visual field is set at 70° by default in Minecraft
    pos = int((side - 1)/2)
    DistanceMax = sqrt(2*pos*pos)
    
    # get the left line
    left = yaw + vf/2
    aLeft = getA(DistanceMax,left,pos)
    bLeft = getB(aLeft,left,pos)

    # get the right line
    right = yaw - vf/2
    aRight = getA(DistanceMax,right,pos)
    bRight = getB(aRight,right,pos)

    # get the minimum and maximum a's
    aMin = int(min(aRight,aLeft))
    aMax = int(max(aRight,aLeft))
    
    # get the distance between the b's
    bDistance = abs(bLeft - bRight)

    # get the positions seen by the agent
    VisualField = {}
    if aMax < pos : 
        if bDistance < side : 
            init = aMin
        elif bDistance >= side :
            init = 0
        for a in range(init, pos+1) :
            bLeft = getB(a,left,pos)
            bRight = getB(a,right,pos)
            VisualField[a] = range(int(min(bLeft,bRight)),int(max(bLeft,bRight))+1)
    elif aMin >= pos :
        if bDistance < side : 
            end = aMax + 1
        elif bDistance >= side :
            end = side + 1
        for a in range(pos, end) :
            bLeft = getB(a,left,pos)
            bRight = getB(a,right,pos)
            VisualField[a] = range(int(min(bLeft,bRight)),int(max(bLeft,bRight))+1)
    elif aMin <= pos and aMax >= pos :
        if left > 90 :
            for a in range(aMin,pos+1) :
                bLeft = getB(a,left,pos)
                bRight = side
                VisualField[a] = range(int(min(bLeft,bRight)),int(max(bLeft,bRight))+1)
            for a in range(pos, aMax) :
                bLeft = side 
                bRight = getB(a,right,pos) 
                VisualField[a] = range(int(min(bLeft,bRight)),int(max(bLeft,bRight))+1)
        elif right < -90 :
            for a in range(aMin,pos+1) :
                bLeft = 0
                bRight = getB(a,right,pos) 
                VisualField[a] = range(int(min(bLeft,bRight)),int(max(bLeft,bRight))+1)
            for a in range(pos, aMax) :
                bLeft = getB(a,left,pos)
                bRight = 0
                VisualField[a] = range(int(min(bLeft,bRight)),int(max(bLeft,bRight))+1)
    return VisualField


# Get the frontal vision, as seen by the agent using the JSON grid and the visual field's coordinates

def GetFrontVision(grid,side,yaw) :
    pos = int((side - 1)/2)
    VisualField = GetVisualField(side,yaw)
    FrontVision = []
    i = 0
    for a in range(side) : 
        tmp = []
        for b in range(side) : 
            if a in VisualField.keys() and b in VisualField[a] :
                tmp.append(grid[i])
            else : 
                tmp.append('no_info')
            i += 1
        FrontVision.append(tmp)
    FrontVision[pos][pos] = "X"
    return FrontVision

def VisualizeFrontVison(FrontVision) :
    for a in FrontVision :
        for b in a :
            print(b,end='\t')
        print("")
    print(" ")


# Get ant's view (Ardin et al, 2016)

def getAntView(h,w,pixels):
    ant_view = []
    id_RGBD = 0
    id_pixels = 0
    while id_pixels < h*w*4 :
        if id_RGBD == 0:
            ant_view.append(pixels[id_pixels]*0.2126)
        elif id_RGBD == 1:
            ant_view[-1] += pixels[id_pixels]*0.7152
        elif id_RGBD == 2:
            ant_view[-1] += pixels[id_pixels]*0.0722
        elif id_RGBD == 3:
            id_RGBD = -1
        id_RGBD += 1
        id_pixels +=1  
    # formula for B&W : Luminance = 0,2126 × Rouge + 0,7152 × Vert + 0,0722 × Bleu

    # resize in image of 10*36 pixels (see Ardin et al)
    ant_view = np.array(ant_view)
    ant_view.shape = (-1, w)
    ant_view = transform.resize(ant_view, (10,36))
    
    # inverse pixel values then equalize the histogram
    ant_view = 1 - ant_view / 255
    ant_view = exposure.equalize_adapthist(ant_view) 

    # normalize the image (see Ardin et al)
    sum_pixels = sqrt(np.sum(ant_view.flatten()))
    ant_view = np.array(list(map(lambda x: x/sum_pixels, ant_view))) # valeurs entières ? ou pas ? 
    
    return ant_view

def visualizeAntVision(ant_view, nb):
    img = Image.fromarray(np.uint8(ant_view*255))
    # img.show()
    img.convert('RGB')
    img.save("./img_random_nav/view_%d.jpg" % nb)


# Get a dictionnary of the objects that can be seen by the agent

def GetObjects(grid,side) :
    pos = (side - 1)/2
    DistanceObjects = []
    num = 1
    aWall = False
    bWall = False
    for a in range(len(grid)) :
        for b in range(len(grid[a])):
            if grid[a][b] == "dirt" :  # detect the wall 
                if ((a < len(grid)-1 and grid[a+1][b] != "dirt" and grid[a+1][b] != "no_info") or (a > 0 and grid[a-1][b] != "dirt" and grid[a-1][b] != "no_info")) and aWall == False :
                    distance = abs(pos - a)
                    angle = degrees(atan2(0,a-pos))
                    DistanceObjects.append({"name": "wall", "type": grid[a][b], "aPos": a, "distance": distance, "angle": angle})
                    aWall = True
                elif ((b < len(grid[a])-1 and grid[a][b+1] != "dirt" and grid[a][b+1] != "no_info") or (b > 0 and grid[a][b-1] != "dirt" and grid[a][b-1] != "no_info")) and bWall == False :
                    distance = abs(pos - b)
                    angle = degrees(atan2(b-pos,0))
                    DistanceObjects.append({"name": "wall", "type": grid[a][b], "bPos": b, "distance": distance, "angle": angle})
                    bWall = True
            elif grid[a][b] != "air" and grid[a][b] != "no_info" and grid[a][b] != "dirt" and grid[a][b] != "X" :
                distance = sqrt((a-pos)*(a-pos)+(b-pos)*(b-pos))
                angle = degrees(atan2((b-pos),(a-pos)))
                obj = {"name": "obj"+str(num), "type": grid[a][b], "aPos": a, "bPos": b, "distance": distance, "angle": angle}
                num += 1
                DistanceObjects.append(obj)
    return DistanceObjects







##############################################################################################################
########################################## NAVIGATION ########################################################
##############################################################################################################

def randomNavigator(movements, last_command):
    movements.remove(last_command)
    return random.choice(movements)








##################################################################################################################
########################################## NEURAL NETWORK ########################################################
##################################################################################################################

def initReactionNetwork():
    begin_time = datetime.datetime.now()
    dt=1.0
    landmark_guidance = Network(dt=dt)

    input_layer = Input(n=360, shape=(10,36))
    PN = LIFNodes(n=360, w=torch.tensor(0.25), tc_decay=10.0)
    KC = LIFNodes(n=20000, traces=True, w=torch.tensor(2.0), tc_decay=10.0)
    EN = LIFNodes(n=1, traces=True, tc_decay=10.0)
    landmark_guidance.add_layer(layer=input_layer, name="Input")
    landmark_guidance.add_layer(layer=PN, name="PN")
    landmark_guidance.add_layer(layer=KC, name="KC")
    landmark_guidance.add_layer(layer=EN, name="EN")

    input_PN = LocalConnection(source=input_layer, target=PN, kernel_size=(10,36), stride=(10,36), n_filters=360)
    PN_KC = Connection(source=PN, target=KC)
    for KC_neuron in range(KC.n):
        idx_PN = list(set(range(PN.n)) - set(np.random.choice(360,10,replace=False)))
        for PN_neuron in idx_PN: 
            PN_KC.w[PN_neuron][KC_neuron] = 0
    KC_EN = Connection(source=KC, target=EN)
    landmark_guidance.add_connection(connection=input_PN, source="Input", target="PN")
    landmark_guidance.add_connection(connection=PN_KC, source="PN", target="KC")
    landmark_guidance.add_connection(connection=KC_EN, source="KC", target="EN")

    input_monitor = Monitor(obj=input_layer, state_vars=("s"))
    PN_monitor = Monitor(obj=PN, state_vars=("s","v"))
    KC_monitor = Monitor(obj=KC, state_vars=("s","v"))
    EN_monitor = Monitor(obj=EN, state_vars=("s","v"))
    landmark_guidance.add_monitor(monitor=input_monitor, name="Input monitor")
    landmark_guidance.add_monitor(monitor=PN_monitor, name="PN monitor")
    landmark_guidance.add_monitor(monitor=KC_monitor, name="KC monitor")
    landmark_guidance.add_monitor(monitor=EN_monitor, name="EN monitor")
    print(datetime.datetime.now()-begin_time)

    return landmark_guidance

def reactionToRandomNavigation(reaction_network, ant_view, plot_option=None, plots=None):
    begin_time = datetime.datetime.now()
    dt = reaction_network.dt
    sim_time = 50 # milliseconds
    reaction_network.reset_state_variables()
    print(ant_view)
    
    input_data = {"Input": torch.from_numpy(np.array([list(map(lambda x: x+np.random.normal(0,0.5), ant_view)) for i in range(int(sim_time/dt))]))}
    reaction_network.run(inputs=input_data, time=sim_time)

    spikes = {
        "Input" : reaction_network.monitors["Input monitor"].get("s"),
        "PN" : reaction_network.monitors["PN monitor"].get("s"),
        "KC" : reaction_network.monitors["KC monitor"].get("s"),
        "EN" : reaction_network.monitors["EN monitor"].get("s")
    }
    voltages = {
        "PN" : reaction_network.monitors["PN monitor"].get("v"),
        "KC" : reaction_network.monitors["KC monitor"].get("v"),
        "EN" : reaction_network.monitors["EN monitor"].get("v")
    }

    if plot_option != None : 
        if plots == None : 
            print("Error : if plot_option is not None, plots has to be defined")
        else :
            plots = plotReactionNetwork(spikes, voltages, plots, plot_option)
            return (spikes, voltages, plots)
    print(datetime.datetime.now() - begin_time)

    return (spikes, voltages)


# def plot_spikes(spikes):
#     n_subplots = len(spikes.keys())
#     spikes = {k: v.view(v.size(0), -1) for (k,v) in spikes.items()}
#     time = spikes[list(spikes.keys())[0]].shape[0]
#     fig, axes = plt.subplots(n_subplots,1)
#     if n_subplots == 1 :
#         axes = [axes]

#     current_subplot = 0
#     for key in spikes.keys():
#         n_neurons = spikes[key].size(-1)
#         time_list = []
#         neuron_list = []
#         for t in range(time):
#             tmp = sum(np.nonzero(spikes[key][t]).tolist(), [])
#             neuron_list += tmp
#             time_list += [t for i in range(len(tmp))]

#         axes[current_subplot].scatter(time_list, neuron_list, s=1, c="cornflowerblue")
#         axes[current_subplot].set_xlim(left=0, right=time)
#         axes[current_subplot].set_ylim(bottom=-0.5, top=n_neurons-0.5)
#         axes[current_subplot].set_title("%s spikes for neurons (%d - %d) from t = %d to %d " % (key, 0, n_neurons, 0, time))
#         plt.setp(axes, xlabel="Simulation time", ylabel="Neuron index")

#         current_subplot += 1
#     plt.tight_layout()

# def plot_voltages(voltage):
#     n_subplots = len(voltage.keys())
#     voltage = {k: v.view(v.size(0), -1) for (k,v) in voltage.items()}
#     time = voltage[list(voltage.keys())[0]].shape[0]
#     fig, axes = plt.subplots(n_subplots,1)
#     if n_subplots == 1 :
#         axes = [axes]

#     current_subplot = 0
#     for key in voltage.keys():
#         n_neurons = voltage[key].size(-1)
#         voltage[key] = voltage[key].t()
        
#         for neuron in voltage[key]:
#             axes[current_subplot].plot(range(1,time+1), neuron)
#         axes[current_subplot].set_title("%s voltages for neurons (%d - %d) from t = %d to %d  " % (key, 0, n_neurons, 0, time))
#         axes[current_subplot].set_xlim(left=0, right=time)
#         plt.setp(axes, xlabel="Simulation time", ylabel="Voltage")

#         current_subplot += 1 
#     plt.tight_layout()

def plotReactionNetwork(spikes, voltages, plots, option, idx=""):
    plt.ioff()
    if len(plots.keys()) == 0:
        im_spikes, axes_spikes = plot_spikes(spikes)
        im_voltage, axes_voltage = plot_voltages(voltages, plot_type="line")
    else : 
        im_spikes, axes_spikes = plot_spikes(spikes, ims=plots["Spikes_ims"], axes=plots["Spikes_axes"])
        im_voltage, axes_voltage = plot_voltages(voltages, plot_type="line", ims=plots["Voltage_ims"], axes=plots["Voltage_axes"])
    
    # plot_spikes(spikes)
    # plot_voltages(voltages)
    
    for (name, item) in [("Spikes_ims",im_spikes), ("Spikes_axes", axes_spikes), ("Voltage_ims", im_voltage), ("Voltage_axes", axes_voltage)]:
        plots[name] = item

    if option == "display":
        plt.show(block=False)
        plt.pause(0.01)
    elif option == "save":
        name_figs = {1: "spikes", 2: "voltage"}
        os.makedirs("./result_random_nav/", exist_ok=True)
        for num in plt.get_fignums():
            plt.figure(num)
            plt.savefig("./result_random_nav/"+name_figs[num]+str(idx)+".png")

    return plots















##############################################################################################################
########################################## RUNNING MISSION ###################################################
##############################################################################################################

agent_host = MalmoPython.AgentHost()
agent_host.setVideoPolicy(MalmoPython.VideoPolicy.KEEP_ALL_FRAMES)
try:
    agent_host.parse( sys.argv )
except RuntimeError as e:
    print('ERROR:',e)
    print(agent_host.getUsage())
    exit(1)
if agent_host.receivedArgument("help"):
    print(agent_host.getUsage())
    exit(0)

# Parameters and network initialization
print("Network initialization", end=' ')
reaction_network = initReactionNetwork()

nb_world_ticks = 0
plots = {}
last_com = "z"

my_mission = MalmoPython.MissionSpec(missionXML, True)
my_mission_record = MalmoPython.MissionRecordSpec()

# Attempt to start a mission:
max_retries = 3
for retry in range(max_retries):
    try:
        # agent_host.startMission( my_mission, my_client_pool, my_mission_record, 0, 42 )
        agent_host.startMission(my_mission, my_mission_record)
        break
    except RuntimeError as e:
        if retry == max_retries - 1:
            print("Error starting mission:",e)
            exit(1)
        else:
            time.sleep(2)

# Loop until mission starts:
print("Waiting for the mission to start ", end=' ')
world_state = agent_host.getWorldState()
while not world_state.has_mission_begun:
    print(".", end="")
    time.sleep(0.1)
    world_state = agent_host.getWorldState()
    for error in world_state.errors:
        print("Error:",error.text)

print()
print("Mission running ", end=' ')

# Loop until mission ends:
while world_state.is_mission_running :
    print(".", end="")
    time.sleep(0.05)
    world_state = agent_host.getWorldState()
    for error in world_state.errors:
        print("Error:",error.text)

    # Observations
    if world_state.number_of_observations_since_last_state > 0 :
        msg = world_state.observations[-1].text 
        ObsJSON = json.loads(msg)
        ObsEnv = {"xPos": ObsJSON["XPos"], "yPos": ObsJSON["YPos"], "zPos": ObsJSON["ZPos"], "yaw": getYaw(-ObsJSON["Yaw"])}

        ### get ant's visions
        ant_view = np.array([getAntView(video_height,video_width,world_state.video_frames[0].pixels)])
        print(ant_view)
        ### launch neural network
        # (spikes, voltages) = reactionToRandomNavigation(reaction_network, ant_view)
        # (spikes, voltages, plots) = reactionToRandomNavigation(reaction_network, ant_view, plot_option="display", plots=plots)

        movements = ["z","q","d"]
        com=randomNavigator(movements, last_com)
        # turn or continue straight on
        if com=="z":
            agent_host.sendCommand("move 1")
            time.sleep(0.25)
        elif com=="q":
            agent_host.sendCommand("turn -1")
            time.sleep(0.25)
            agent_host.sendCommand("turn 0")
        elif com=="d":
            agent_host.sendCommand("turn 1")
            time.sleep(0.25)
            agent_host.sendCommand("turn 0")
        # move for 1 world tick
        agent_host.sendCommand("move 1")
        last_com = com

    nb_world_ticks += 1

print("Simulation over")

# plotReactionNetwork(spikes, voltages, plots, option="save")

print()
print("Mission ended")
# Mission has ended.