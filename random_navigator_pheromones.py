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
from lxml import etree
import datetime

import torch
from bindsnet.network import Network
from bindsnet.network.nodes import Input
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from AntLearning import STDP, AllToAllConnection, Izhikevich
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

timeRandomNav = 100 # time during which agent is randomly walking around, leaving pheromones behind him (at the end of that, turn around and try to follow the pheromones) 

# Create random objects in the arena

def GenRandomObject(xPos, zPos, ArenaSide, ArenaFloor, root):
    print("Waiting for the world to load ", end=' ')
    # we want 2 obstacles in a 10cm*10cm region => for 1cm*1cm = 1block*1block, we need 0.05obstacles
    nb = int(0.01*ArenaSide*ArenaSide)
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
            etree.SubElement(root, "DrawBlock", type=BlockType, x=str(x), y=str(y), z=str(z))
            PreviousCoord.append([x,z])
    print()





#########################################################################################################
############################################# XML #######################################################
#########################################################################################################

def GetMissionXML(ArenaSide, ArenaFloor, xPos, zPos, video_height, video_width):
    AntWorld = etree.parse("ant_world.xml") # returns ElementTree object

    DrawingDecorator = AntWorld.xpath("/Mission/ServerSection/ServerHandlers/DrawingDecorator")[0]

    AirCuboid = DrawingDecorator.getchildren()[0]
    AirCuboid.attrib['x1'] = str(old_div(-ArenaSide,2))
    AirCuboid.attrib["y1"] = str(ArenaFloor)
    AirCuboid.attrib["z1"] = str(old_div(-ArenaSide,2))
    AirCuboid.attrib["x2"] = str(old_div(ArenaSide,2))
    AirCuboid.attrib["z2"] = str(old_div(ArenaSide,2))

    GenRandomObject(xPos, zPos, ArenaSide, ArenaFloor, DrawingDecorator)

    AgentStartPlacement = AntWorld.xpath("/Mission/AgentSection/AgentStart/Placement")[0]
    AgentStartPlacement.attrib["x"] = str(xPos)
    AgentStartPlacement.attrib["y"] = str(ArenaFloor)
    AgentStartPlacement.attrib["z"] = str(zPos)

    VideoProducer = AntWorld.xpath("/Mission/AgentSection/AgentHandlers/VideoProducer")[0]
    variables_VideoProducer = {"Width": str(video_width), "Height": str(video_height)}
    for child in VideoProducer.getchildren():
        child.text = variables_VideoProducer[child.tag]
    
    return etree.tostring(AntWorld)


missionXML = GetMissionXML(ArenaSide, ArenaFloor, xPos, zPos, video_height, video_width)







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
    
    # resize in image of 10*39 pixels (see Ardin et al)
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

def randomNavigator(ObsEnv, direction, last_command):
    BlocksInFront = [ObsEnv["GridObstacles"][i] for i in getWhatIsInFront(round(ObsEnv['yaw']), grid="GridObstacles")[1:-1]]
    if last_command == None : 
        return random.choice(direction)
    elif BlocksInFront == ['air','air','air']:
        return "z"
    elif BlocksInFront != ['air','air','air']:  # tourner pour s'éloigner de l'obstacle ? 
        if BlocksInFront[0] == "air":
            return "q"
        elif BlocksInFront[2] == "air":
            return "d"
        else : 
            return random.choice(direction)

def addPheromones(ObsEnv,agent,turn=False):
    if turn==False:
        xVar1 = random.randint(0,2) 
        xVar2 = random.randint(0,2)+1
        zVar1 = random.randint(0,2)
        zVar2 = random.randint(0,2)+1
        pheromonesCoord = [[x,z] for x in range(round(ObsEnv["xPos"])-xVar1, round(ObsEnv["xPos"])+xVar2) for z in range(round(ObsEnv["zPos"])-zVar1, round(ObsEnv["zPos"])+zVar2)]
    elif turn==True:
        pheromonesCoord = [[x,z] for x in range(round(ObsEnv["xPos"])-1, round(ObsEnv["xPos"])+2) for z in range(round(ObsEnv["zPos"])-1, round(ObsEnv["zPos"])+2)]
    for [x,z] in pheromonesCoord:
        agent.sendCommand('chat /setblock ' + str(x) + ' ' + str(ArenaFloor-1) + ' ' + str(z) + ' gold_block')

def getWhatIsInFront(agentYaw, grid):
    if grid == "GridObstacles":
        whatIsInFront = {
            # for PheromonesTrace of dimension 3*3 and for GridObstacles
            range(-180, -157): [3,0,1,2,5], # 1/2 of north square => centered on index=1 of PheromonesTrace grid
            range(-157, -112): [6,3,0,1,2], # north-west square => centered on index=0 of PheromonesTrace grid
            range(-112, -67): [7,6,3,0,1], # west block => centered on index=3 of PheromonesTrace grid
            range(-67, -22): [8,7,6,3,0], # south-west square => centered on index=6 of PheromonesTrace grid        
            range(-22, 22): [5,8,7,6,3], # south square => centered on index=7 of PheromonesTrace grid
            range(22, 67): [2,5,8,7,6], # south-est square => centered on index=8 of PheromonesTrace grid
            range(67, 112): [1,2,5,8,7], # est square => centered on index=5 of PheromonesTrace grid
            range(112, 157): [0,1,2,5,8], # north-est square => centered on index=2 of PheromonesTrace grid
            range(157, 181): [3,0,1,2,5] # 1/2 of north square => centered on index=1 of PheromonesTrace grid
        }
    elif grid == "GridPheromonesTrace" :
        whatIsInFront = {    
            # for PheromonesTrace of dimension 5*5
            range(-180, -157): [5,10,11, 0,6, 1,2,7,3, 4,8, 9,13,14], # 1/2 of north square => centered on index=2 of PheromonesTrace grid
            range(-157, -112): [15,16,20, 10,11, 1,0,6,5, 2,7, 3,4,8], # north-west square => centered on index=0 of PheromonesTrace grid
            range(-112, -67): [21,22,17, 20,16, 5,10,11,15, 0,6, 1,2,7], # west block => centered on index=10 of PheromonesTrace grid
            range(-67, -22): [18,23,24, 17,22, 15,16,20,21, 10,11, 0,5,6], # south-west square => centered on index=20 of PheromonesTrace grid        
            range(-22, 22): [13,14,19, 18,24, 21,17,22,23, 20,16, 10,11,15], # south square => centered on index=22 of PheromonesTrace grid
            range(22, 67): [4,8,9, 13,14, 19,18,24,23, 17,22, 16,20,21], # south-est square => centered on index=24 of PheromonesTrace grid
            range(67, 112): [2,3,7, 4,8, 9,13,14,19, 18,24, 17,22,23], # est square => centered on index=14 of PheromonesTrace grid
            range(112, 157): [0,1,6, 2,7, 3,4,8,9, 13,14, 18,19,24], # north-est square => centered on index=4 of PheromonesTrace grid
            range(157, 181): [5,10,11, 0,6, 1,2,7,3, 4,8, 9,13,14] # 1/2 of north square => centered on index=2 of PheromonesTrace grid
        }    
    for yaw in whatIsInFront.keys():
        if agentYaw in yaw:
            # print("YESSSSS", yaw, "=>", whatIsInFront[yaw])
            return whatIsInFront[yaw]

def followPheromonesPath(ObsEnv,agent):
    BlocksInFront = getWhatIsInFront(round(ObsEnv["yaw"]), grid="GridPheromonesTrace")
    Obstacles = getWhatIsInFront(round(ObsEnv["yaw"]), grid="GridObstacles")

    directions = {
        # for PheromonesTrace of dimension 3*3
        # BlocksInFront[0]: "q",
        # BlocksInFront[1]: "q",
        # BlocksInFront[2]: "z",
        # BlocksInFront[3]: "d",
        # BlocksInFront[4]: "d"
        # for PheromonesTrace of dimension 5*5
        BlocksInFront[0]: "q",
        BlocksInFront[1]: "q",
        BlocksInFront[2]: "q",
        BlocksInFront[3]: "qz",
        BlocksInFront[4]: "qz",
        BlocksInFront[5]: "z",
        BlocksInFront[6]: "z",
        BlocksInFront[7]: "z",
        BlocksInFront[8]: "z",
        BlocksInFront[9]: "zd",
        BlocksInFront[10]: "zd",
        BlocksInFront[11]: "d",
        BlocksInFront[12]: "d",
        BlocksInFront[13]: "d",
    }
    BlocksWithGold = list(filter(lambda x: ObsEnv["GridPheromonesTrace"][x] == 'gold_block', BlocksInFront))
    print(BlocksWithGold)
    print(Obstacles[2])
    print(Obstacles)
    print(ObsEnv["GridObstacles"][Obstacles[2]])
    if BlocksWithGold == [] :
        return random.choice(["q","d"])
    elif BlocksInFront[6] in BlocksWithGold and BlocksInFront[7] in BlocksWithGold and ObsEnv["GridObstacles"][Obstacles[2]] == "air":
        return "z"
    
    nb_BlocksInFront = {
        "q": 0,
        "qz": 0,
        "z": 0,
        "zd": 0,
        "d": 0
    }
    for idx in BlocksWithGold:
        nb_BlocksInFront[directions[idx]] += 1
    print(nb_BlocksInFront)

    if ((nb_BlocksInFront["qz"] + nb_BlocksInFront["z"] > nb_BlocksInFront["q"]) or (nb_BlocksInFront["zd"] + nb_BlocksInFront["z"] > nb_BlocksInFront["d"])) and Obstacles[2] == "air":
        return "z"
    elif nb_BlocksInFront["qz"] + nb_BlocksInFront["q"] > nb_BlocksInFront["zd"] + nb_BlocksInFront["d"]:
        return "q"
    elif nb_BlocksInFront["qz"] + nb_BlocksInFront["q"] < nb_BlocksInFront["zd"] + nb_BlocksInFront["d"]:
        return "d"
    else :
        return random.choice(["q","d"])

    # for PheromonesTrace of dimension 3*3
    # elif BlocksInFront[2] in BlocksWithGold:
    #     return directions[BlocksInFront[2]]
    # else:
    #     return directions[random.choice(BlocksWithGold)]


def AgentTurn(com):
    if com=="q":
        agent_host.sendCommand("turn -1")
        time.sleep(0.25)
        agent_host.sendCommand("turn 0")
    elif com=="d":
        agent_host.sendCommand("turn 1")
        time.sleep(0.25)
        agent_host.sendCommand("turn 0")










##################################################################################################################
########################################## NEURAL NETWORK ########################################################
##################################################################################################################

def initReactionNetwork(
    dt=1.0,
    A=1.0,
    PN_KC_weight=0.25,
    KC_EN_weight=2.0,
    min_weight=0.0001,
    PN_thresh=-40.0,
    KC_thresh=-25.0,
    EN_thresh=-40.0,
    ):
    
    begin_time = datetime.datetime.now()
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

    return landmark_guidance

def reactionToRandomNavigation(
    reaction_network, 
    ant_view, 
    phase, # choice between 3 phases : reaction (no learning, equivalent phase 1), learning (with learning, equivalent phase 2), test (after learning, equivalent phase 3)
    BA=0.5,
    exposition_time=40, # milliseconds => how long the image is presented to the network
    simulation_time=50, # milliseconds => how long the simulation runs 
    modification=5250.0,  # best results with 1.0 for CurrentLIF (for LIF, 0.1) => augmented in order to encode the richess of the input
    plot_option=None, 
    plots=None):

    reaction_network.reset_state_variables()
    begin_time = datetime.datetime.now()
    dt = reaction_network.dt
    
    input_data = {"Input": torch.from_numpy(modification * np.array([ant_view if i <= exposition_time else np.zeros((1,10,36)) for i in range(int(simulation_time/dt))]))}

    if phase in ["reaction","test"]:
        reaction_network.learning = False
    elif phase == "learning":
        reaction_network.learning = True

    reaction_network.run(inputs=input_data, time=simulation_time, reward=BA, n_timesteps=simulation_time/dt)

    spikes = {
        "PN" : reaction_network.monitors["PN monitor"].get("s"),
        "KC" : reaction_network.monitors["KC monitor"].get("s"),
        "EN" : reaction_network.monitors["EN monitor"].get("s")
    }
    voltages = {
        "PN" : reaction_network.monitors["PN monitor"].get("v"),
        "KC" : reaction_network.monitors["KC monitor"].get("v"),
        "EN" : reaction_network.monitors["EN monitor"].get("v")
    }

    nb_spikes_EN = len(torch.nonzero(spikes["EN"]))

    if plot_option != None : 
        if plots == None : 
            print("Error : if plot_option is not None, plots has to be defined")
        else :
            plots = plotReactionNetwork(spikes, voltages, plots, plot_option)
            return (nb_spikes_EN, plots)
    print(datetime.datetime.now() - begin_time)

    return nb_spikes_EN


def plotReactionNetwork(spikes, voltages, plots, option, idx=""):
    plt.ioff()
    if len(plots.keys()) == 0:
        im_spikes, axes_spikes = plot_spikes(spikes)
        im_voltage, axes_voltage = plot_voltages(voltages, plot_type="line")
    else : 
        im_spikes, axes_spikes = plot_spikes(spikes, ims=plots["Spikes_ims"], axes=plots["Spikes_axes"])
        im_voltage, axes_voltage = plot_voltages(voltages, plot_type="line", ims=plots["Voltage_ims"], axes=plots["Voltage_axes"])
    
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

path_StartToEnd = True
path_EndToStart = False

my_mission = MalmoPython.MissionSpec(missionXML, False)
my_mission_record = MalmoPython.MissionRecordSpec()

# Attempt to start a mission:
max_retries = 3
for retry in range(max_retries):
    try:
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
    path_start = False # indicates wether the agent has reached the beginning (diamond blocks tower) of its path after at least one loop
    path_end = False # indicates wether the agent has reached the end of its path (emerauld blocks tower) after at least one loop

    print(".", end="")
    time.sleep(0.1)
    world_state = agent_host.getWorldState()
    for error in world_state.errors:
        print("Error:",error.text)
    for reward in world_state.rewards:
        if reward.getValue() == 100.0:
            path_start = True
        elif reward.getValue() == -100.0:
            path_end = True

    # Observations
    if world_state.number_of_observations_since_last_state > 0 :
        msg = world_state.observations[-1].text 
        ObsJSON = json.loads(msg)
        ObsEnv = {"xPos": ObsJSON["XPos"], "yPos": ObsJSON["YPos"], "zPos": ObsJSON["ZPos"], "yaw": getYaw(-ObsJSON["Yaw"])}
        ObsEnv["GridPheromonesTrace"] = ObsJSON["PheromonesTrace"] # permet de récupérer ce que l'agent perçoit au niveau des phéromones
        ObsEnv["GridObstacles"] = ObsJSON["Obstacles"] # permet de dire si l'agent s'approche d'un obstacle

        
        direction = ["q","d"]
        if nb_world_ticks < timeRandomNav:
            addPheromones(ObsEnv,agent_host) 
            com=randomNavigator(ObsEnv, direction,last_com)
            if nb_world_ticks % 20 == 0:
                ### get ant's vision
                ant_view = np.array([getAntView(video_height,video_width,world_state.video_frames[0].pixels)])
                reactionToRandomNavigation(reaction_network, ant_view, phase="learning")

        elif nb_world_ticks == timeRandomNav:
            addPheromones(ObsEnv,agent_host,turn=True)
            print("Follow the pheromones path")
            
            # add a column inside the arena at (x,y,z) original of the agent : when it reaches the colum, it has reached its point of departure and learned the views along the way
            for height in range(ArenaFloor, ArenaFloor+4):
                agent_host.sendCommand('chat /setblock ' + str(xPos) + ' ' + str(height) + ' ' + str(zPos) + ' diamond_block')
            
            # get the current location
            path_end = {"x": ObsEnv["xPos"], "z": ObsEnv["zPos"]}

            # faces the other way
            agent_host.sendCommand("move 0")
            agent_host.sendCommand("turn -1")
            time.sleep(1)
            agent_host.sendCommand("turn 0")

            path_StartToEnd = False
            path_EndToStart = True
        
        elif nb_world_ticks > timeRandomNav : 
            
            if path_EndToStart == True :
                com = followPheromonesPath(ObsEnv, agent_host)

            elif path_start == True :  # the agent just reached the beginning of the path (diamond blocks tour) and will try to find its path using the views it learned
                for height in range(ArenaFloor, ArenaFloor+4):  
                    agent_host.sendCommand('chat /setblock ' + str(path_end["x"]) + ' ' + str(height) + ' ' + str(path_end["z"]) + ' emerauld_block')
            
                # faces the other way
                agent_host.sendCommand("move 0")
                agent_host.sendCommand("turn -1")
                time.sleep(1)
                agent_host.sendCommand("turn 0")

                path_StartToEnd = True
                path_EndToStart = False
            
            elif path_StartToEnd == True :  # find most familar views : the gaent stops and compare 3 different views
                most_familiar_view = {"EN": None, "direction": None}
                turn_to_test_views = {"right": "q", "ahead": "d", "left": "d"}

                for direction in ["right","ahead","left"]:  
                    AgentTurn(turn_to_test_views[direction])  # turn once to the right, then turn twice to the left, each time to get ant's vision
                    ant_view = np.array([getAntView(video_height,video_width,world_state.video_frames[0].pixels)])
                    nb_spikes_EN = reactionToRandomNavigation(reaction_network, ant_view, phase="test")
                    
                    if most_familiar_view["EN"] == None or most_familiar_view["EN"] > nb_spikes_EN:
                        most_familiar_view["EN"] = nb_spikes_EN
                        most_familiar_view["direction"] = direction
                
                AgentTurn("q") # agent returns in original axe
                turn_to_most_familar_view = {"right": "q", "ahead": "z", "left": "d"} 
                com = turn_to_most_familar_view[most_familiar_view["direction"]]

            elif path_end == True :
                agent_host.sendCommand("move 0")
                agent_host.sendCommand("turn -1")
                time.sleep(1)
                agent_host.sendCommand("turn 0")
            

        # turn 
        AgentTurn(com)

        agent_host.sendCommand("move 1")
        time.sleep(0.25)
        agent_host.sendCommand("move 0")

        nb_world_ticks += 1
        last_com = com

print()
print("Mission ended")
# Mission has ended.