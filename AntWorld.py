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




###############################################################################################################################################
####################################################### WORLD INITIALISATION ##################################################################
###############################################################################################################################################

class World():

    def __init__(
        self,
        ArenaSide, # the agent's size is 1 block = 1 cm (for an ant) => the agent evolves in an arena of diameter 10m*10m
        ArenaFloor,
        xPos,
        zPos,
        video_height,
        video_width,
        timeRandomNav # time during which agent is randomly walking around, leaving pheromones behind him (at the end of that, turn around and try to follow the pheromones) 
        ):

        self.ArenaSide = ArenaSide
        self.ArenaFloor = ArenaFloor
        self.xPos = xPos
        self.zPos = zPos
        
        self.video_height = video_height
        self.video_width = video_width

        self.timeRandomNav = timeRandomNav


    # Create random objects in the arena

    def GenRandomObject(
        self,
        root):

        print("Waiting for the world to load ", end=' ')

        # we want 2 obstacles in a 10cm*10cm region => for 1cm*1cm = 1block*1block, we need 0.05obstacles
        nb = int(0.01*self.ArenaSide*self.ArenaSide)
        BlockType = "redstone_block"
        ObjHeight = [2,3,4,5]
        ObjWidth = [1,2]
        max = old_div(self.ArenaSide,2)
        PreviousCoord = [[x,z] for x in range(self.xPos-1, self.xPos+2) for z in range(self.zPos-1, self.zPos+2)]

        for i in range(nb) :
            if i % 50 == 0 :
                print(".", end="")
            height = random.choice(ObjHeight)
            width = random.choice(ObjWidth)
            [x0,z0] = [self.xPos, self.zPos]
        
            while [x0,z0] in PreviousCoord :
                x0 = random.randint(-max,max)
                z0 = random.randint(-max,max)
        
            for [x,y,z] in [[x0+x, self.ArenaFloor+y, z0+z] for x in range(width) for y in range(height) for z in range(width)]:
                etree.SubElement(root, "DrawBlock", type=BlockType, x=str(x), y=str(y), z=str(z))
                PreviousCoord.append([x,z])
        
        print()


    def GetMissionXML(self):
        AntWorld = etree.parse("ant_world.xml") # returns ElementTree object

        DrawingDecorator = AntWorld.xpath("/Mission/ServerSection/ServerHandlers/DrawingDecorator")[0]

        # arena's initialisation
        AirCuboid = DrawingDecorator.getchildren()[0]
        AirCuboid.attrib['x1'] = str(old_div(-self.ArenaSide,2))
        AirCuboid.attrib["y1"] = str(self.ArenaFloor)
        AirCuboid.attrib["z1"] = str(old_div(-self.ArenaSide,2))
        AirCuboid.attrib["x2"] = str(old_div(self.ArenaSide,2))
        AirCuboid.attrib["z2"] = str(old_div(self.ArenaSide,2))

        # add obstacles
        self.GenRandomObject(DrawingDecorator)

        # agent's initialisation
        AgentStartPlacement = AntWorld.xpath("/Mission/AgentSection/AgentStart/Placement")[0]
        AgentStartPlacement.attrib["x"] = str(self.xPos)
        AgentStartPlacement.attrib["y"] = str(self.ArenaFloor)
        AgentStartPlacement.attrib["z"] = str(self.zPos)

        # video producer's initialisation
        VideoProducer = AntWorld.xpath("/Mission/AgentSection/AgentHandlers/VideoProducer")[0]
        variables_VideoProducer = {"Width": str(self.video_width), "Height": str(self.video_height)}
        for child in VideoProducer.getchildren():
            child.text = variables_VideoProducer[child.tag]
        
        return etree.tostring(AntWorld)








##############################################################################################################
########################################## NAVIGATION ########################################################
##############################################################################################################

class Navigation():

    def __init__(
        self,
        agent_host,
        ArenaFloor
        ):
        
        self.agent_host = agent_host
        self.ArenaFloor = ArenaFloor

    def randomNavigator(
        self,
        ObsEnv, 
        direction, 
        last_command):
        
        BlocksInFront = [ObsEnv["GridObstacles"][i] for i in self.getWhatIsInFront(round(ObsEnv['yaw']), grid="GridObstacles")[1:-1]]
        if last_command == None : 
            return random.choice(direction)
        elif BlocksInFront == ['air','air','air']:
            return "ahead"
        elif BlocksInFront != ['air','air','air']:  # tourner pour s'éloigner de l'obstacle ? 
            if BlocksInFront[0] == "air":
                return "left"
            elif BlocksInFront[2] == "air":
                return "right"
            else : 
                return random.choice(direction)


    def addPheromones(
        self,
        ObsEnv,
        agent,turn=False):

        if turn==False:
            xVar1 = random.randint(0,2) 
            xVar2 = random.randint(0,2)+1
            zVar1 = random.randint(0,2)
            zVar2 = random.randint(0,2)+1
            pheromonesCoord = [[x,z] for x in range(round(ObsEnv["xPos"])-xVar1, round(ObsEnv["xPos"])+xVar2) for z in range(round(ObsEnv["zPos"])-zVar1, round(ObsEnv["zPos"])+zVar2)]
        elif turn==True:
            pheromonesCoord = [[x,z] for x in range(round(ObsEnv["xPos"])-1, round(ObsEnv["xPos"])+2) for z in range(round(ObsEnv["zPos"])-1, round(ObsEnv["zPos"])+2)]
        for [x,z] in pheromonesCoord:
            agent.sendCommand('chat /setblock ' + str(x) + ' ' + str(self.ArenaFloor-1) + ' ' + str(z) + ' gold_block')


    def getWhatIsInFront(
        self,
        agentYaw, 
        grid):
        
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
                return whatIsInFront[yaw]


    def followPheromonesPath(
        self,
        ObsEnv,
        agent):
        
        BlocksInFront = self.getWhatIsInFront(round(ObsEnv["yaw"]), grid="GridPheromonesTrace")
        Obstacles = self.getWhatIsInFront(round(ObsEnv["yaw"]), grid="GridObstacles")

        directions = {
            BlocksInFront[0]: "left",
            BlocksInFront[1]: "left",
            BlocksInFront[2]: "left",
            BlocksInFront[3]: "left-ahead",
            BlocksInFront[4]: "left-ahead",
            BlocksInFront[5]: "ahead",
            BlocksInFront[6]: "ahead",
            BlocksInFront[7]: "ahead",
            BlocksInFront[8]: "ahead",
            BlocksInFront[9]: "ahead-right",
            BlocksInFront[10]: "ahead-right",
            BlocksInFront[11]: "right",
            BlocksInFront[12]: "right",
            BlocksInFront[13]: "right",
        }
        BlocksWithGold = list(filter(lambda x: ObsEnv["GridPheromonesTrace"][x] == 'gold_block', BlocksInFront))
        if BlocksWithGold == [] :
            return random.choice(["left","right"])
        elif BlocksInFront[6] in BlocksWithGold and BlocksInFront[7] in BlocksWithGold and ObsEnv["GridObstacles"][Obstacles[2]] == "air":
            return "ahead"
        
        nb_BlocksInFront = {
            "left": 0,
            "left-ahead": 0,
            "ahead": 0,
            "ahead-right": 0,
            "right": 0
        }
        for idx in BlocksWithGold:
            nb_BlocksInFront[directions[idx]] += 1

        if ((nb_BlocksInFront["left-ahead"] + nb_BlocksInFront["ahead"] > nb_BlocksInFront["left"]) or (nb_BlocksInFront["ahead-right"] + nb_BlocksInFront["ahead"] > nb_BlocksInFront["right"])) and Obstacles[2] == "air":
            return "ahead"
        elif nb_BlocksInFront["left-ahead"] + nb_BlocksInFront["left"] > nb_BlocksInFront["ahead-right"] + nb_BlocksInFront["right"]:
            return "left"
        elif nb_BlocksInFront["left-ahead"] + nb_BlocksInFront["left"] < nb_BlocksInFront["ahead-right"] + nb_BlocksInFront["right"]:
            return "right"
        else :
            return random.choice(["left","right"])


    def AgentMove(
        self,
        com):
        
        if com=="left":
            self.agent_host.sendCommand("turn -1")
            time.sleep(0.25)
            self.agent_host.sendCommand("turn 0")
        elif com=="right":
            self.agent_host.sendCommand("turn 1")
            time.sleep(0.25)
            self.agent_host.sendCommand("turn 0")
        elif com=="ahead":
            self.agent_host.sendCommand("move 1")
            time.sleep(0.25)
            self.agent_host.sendCommand("move 0")
        elif com=="U-turn":
            self.agent_host.sendCommand("move 0")
            self.agent_host.sendCommand("turn -1")
            time.sleep(1)
            self.agent_host.sendCommand("turn 0")









###########################################################################################################################################################
################################################################## GET SENSORY INFORMATION ################################################################
###########################################################################################################################################################

class SensoryInformation():

    def __init__(self):
        pass

    # Get the correct Yaw (strictly between 180 and -180) at each time

    def getYaw(self, yaw) :
        if yaw / 180 > 1 :
            yaw = yaw - 360
        elif yaw / 180 < -1 :
            yaw = yaw + 360
        return yaw


    # Get the coordinates corresponding to the visual field

    def getA(self, lenght, angle, pos) :
        return lenght * cos(radians(angle)) + pos

    def getB(self, a, angle, pos):
        return (a-pos) * tan(radians(angle)) + pos


    def getVisualField(self, side,yaw,vf=70) :
        # visual field is set at 70° by default in Minecraft
        pos = int((side - 1)/2)
        DistanceMax = sqrt(2*pos*pos)
        
        # get the left line
        left = yaw + vf/2
        aLeft = self.getA(DistanceMax,left,pos)
        bLeft = self.getB(aLeft,left,pos)

        # get the right line
        right = yaw - vf/2
        aRight = self.getA(DistanceMax,right,pos)
        bRight = self.getB(aRight,right,pos)

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
                bLeft = self.getB(a,left,pos)
                bRight = self.getB(a,right,pos)
                VisualField[a] = range(int(min(bLeft,bRight)),int(max(bLeft,bRight))+1)
        elif aMin >= pos :
            if bDistance < side : 
                end = aMax + 1
            elif bDistance >= side :
                end = side + 1
            for a in range(pos, end) :
                bLeft = self.getB(a,left,pos)
                bRight = self.getB(a,right,pos)
                VisualField[a] = range(int(min(bLeft,bRight)),int(max(bLeft,bRight))+1)
        elif aMin <= pos and aMax >= pos :
            if left > 90 :
                for a in range(aMin,pos+1) :
                    bLeft = self.getB(a,left,pos)
                    bRight = side
                    VisualField[a] = range(int(min(bLeft,bRight)),int(max(bLeft,bRight))+1)
                for a in range(pos, aMax) :
                    bLeft = side 
                    bRight = self.getB(a,right,pos) 
                    VisualField[a] = range(int(min(bLeft,bRight)),int(max(bLeft,bRight))+1)
            elif right < -90 :
                for a in range(aMin,pos+1) :
                    bLeft = 0
                    bRight = self.getB(a,right,pos) 
                    VisualField[a] = range(int(min(bLeft,bRight)),int(max(bLeft,bRight))+1)
                for a in range(pos, aMax) :
                    bLeft = self.getB(a,left,pos)
                    bRight = 0
                    VisualField[a] = range(int(min(bLeft,bRight)),int(max(bLeft,bRight))+1)
        return VisualField


    # Get the frontal vision, as seen by the agent using the JSON grid and the visual field's coordinates

    def getFrontVision(self, grid,side,yaw) :
        pos = int((side - 1)/2)
        VisualField = self.getVisualField(side,yaw)
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

    def visualizeFrontVison(self, FrontVision) :
        for a in FrontVision :
            for b in a :
                print(b,end='\t')
            print("")
        print(" ")


    # Get ant's view (Ardin et al, 2016)

    def getAntView(self, h,w,pixels):
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

    def visualizeAntVision(self, ant_view, nb):
        img = Image.fromarray(np.uint8(ant_view*255))
        # img.show()
        img.convert('RGB')
        img.save("./img_random_nav/view_%d.jpg" % nb)


    # Get a dictionnary of the objects that can be seen by the agent

    def getObjects(self, grid,side) :
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









#######################################################################################################################################################
################################################################## NEURAL NETWORK #####################################################################
#######################################################################################################################################################

class AutonomousAgent():

    def __init__(
        self,
        dt=1.0,
        A=1.0,
        PN_KC_weight=0.25,
        KC_EN_weight=2.0,
        min_weight=0.0001,
        PN_thresh=-40.0,
        KC_thresh=-25.0,
        EN_thresh=-40.0,
        ):

        self.landmark_guidance = Network(dt=dt)

        # layers
        self.input_layer = Input(n=360, shape=(10,36))
        self.PN = Izhikevich(n=360, traces=True, tc_decay=10.0, thresh=PN_thresh, rest=-60.0, C=100, a=0.3, b=-0.2, c=-65, d=8, k=2)
        self.KC = Izhikevich(n=20000, traces=True, tc_decay=10.0, thresh=KC_thresh, rest=-85.0, C=4, a=0.01, b=-0.3, c=-65, d=8, k=0.035)
        self.EN = Izhikevich(n=1, traces=True, tc_decay=10.0, thresh=EN_thresh, rest=-60.0, C=100, a=0.3, b=-0.2, c=-65, d=8, k=2)
        self.landmark_guidance.add_layer(layer=self.input_layer, name="Input")
        self.landmark_guidance.add_layer(layer=self.PN, name="PN")
        self.landmark_guidance.add_layer(layer=self.KC, name="KC")
        self.landmark_guidance.add_layer(layer=self.EN, name="EN")

        # connections
        connection_weight = torch.zeros(self.input_layer.n, self.PN.n).scatter_(1,torch.tensor([[i,i] for i in range(self.PN.n)]),1.)
        self.input_PN = Connection(source=self.input_layer, target=self.PN, w=connection_weight)

        connection_weight = torch.zeros(PN.n, KC.n).t()
        connection_weight = connection_weight.scatter_(1, torch.tensor([np.random.choice(PN.n, size=10, replace=False) for i in range(KC.n)]).long(), PN_KC_weight)
        self.PN_KC = AllToAllConnection(source=self.PN, target=self.KC, w=connection_weight.t(), tc_synaptic=3.0, phi=0.93)

        self.KC_EN = AllToAllConnection(source=self.KC, target=self.EN, w=torch.ones(self.KC.n, self.EN.n)*2.0, tc_synaptic=8.0, phi=8.0)

        self.landmark_guidance.add_connection(connection=self.input_PN, source="Input", target="PN")
        self.landmark_guidance.add_connection(connection=self.PN_KC, source="PN", target="KC")
        self.landmark_guidance.add_connection(connection=self.KC_EN, source="KC", target="EN")

        # learning rule
        self.KC_EN.update_rule = STDP(connection=self.KC_EN, nu=(-A,-A), tc_eligibility_trace=40.0, tc_plus=15, tc_minus=15, tc_reward=20.0, min_weight=min_weight)

        # monitors
        input_monitor = Monitor(obj=self.input_layer, state_vars=("s"))
        PN_monitor = Monitor(obj=self.PN, state_vars=("s","v"))
        KC_monitor = Monitor(obj=self.KC, state_vars=("s","v"))
        EN_monitor = Monitor(obj=self.EN, state_vars=("s","v"))
        self.landmark_guidance.add_monitor(monitor=input_monitor, name="Input monitor")
        self.landmark_guidance.add_monitor(monitor=PN_monitor, name="PN monitor")
        self.landmark_guidance.add_monitor(monitor=KC_monitor, name="KC monitor")
        self.landmark_guidance.add_monitor(monitor=EN_monitor, name="EN monitor")

        # plots
        self.plots = {}


        # number of EN spikes during the simulation
        self.nb_spikes_EN = 0


    def reactionToRandomNavigation(
        self,
        ant_view, 
        phase, # choice between 3 phases : reaction (no learning, equivalent phase 1), learning (with learning, equivalent phase 2), test (after learning, equivalent phase 3)
        BA=0.5,
        exposition_time=40, # milliseconds => how long the image is presented to the network
        simulation_time=50, # milliseconds => how long the simulation runs 
        modification=5250.0,  # best results with 1.0 for CurrentLIF (for LIF, 0.1) => augmented in order to encode the richess of the input
        plot_option=None):

        self.landmark_guidance.reset_state_variables()
        dt = self.landmark_guidance.dt
        
        input_data = {"Input": torch.from_numpy(modification * np.array([ant_view if i <= exposition_time else np.zeros((1,10,36)) for i in range(int(simulation_time/dt))]))}

        if phase in ["reaction","test"]:
            self.landmark_guidance.learning = False
        elif phase == "learning":
            self.landmark_guidance.learning = True

        self.landmark_guidance.run(inputs=input_data, time=simulation_time, reward=BA, n_timesteps=simulation_time/dt)

        self.spikes = {
            "PN" : self.landmark_guidance.monitors["PN monitor"].get("s"),
            "KC" : self.landmark_guidance.monitors["KC monitor"].get("s"),
            "EN" : self.landmark_guidance.monitors["EN monitor"].get("s")
        }
        self.voltages = {
            "PN" : self.landmark_guidance.monitors["PN monitor"].get("v"),
            "KC" : self.landmark_guidance.monitors["KC monitor"].get("v"),
            "EN" : self.landmark_guidance.monitors["EN monitor"].get("v")
        }

        self.nb_spikes_EN = len(torch.nonzero(self.spikes["EN"]))

        if plot_option != None : 
            self.plotReactionNetwork(self.plots, plot_option)



    def plotReactionNetwork(
        self,
        option, 
        idx=""):
        plt.ioff()
        if len(self.plots.keys()) == 0:
            im_spikes, axes_spikes = plot_spikes(self.spikes)
            im_voltage, axes_voltage = plot_voltages(self.voltages, plot_type="line")
        else : 
            im_spikes, axes_spikes = plot_spikes(self.spikes, ims=self.plots["Spikes_ims"], axes=self.plots["Spikes_axes"])
            im_voltage, axes_voltage = plot_voltages(self.voltages, plot_type="line", ims=self.plots["Voltage_ims"], axes=self.plots["Voltage_axes"])
        
        for (name, item) in [("Spikes_ims",im_spikes), ("Spikes_axes", axes_spikes), ("Voltage_ims", im_voltage), ("Voltage_axes", axes_voltage)]:
            self.plots[name] = item

        if option == "display":
            plt.show(block=False)
            plt.pause(0.01)
        elif option == "save":
            name_figs = {1: "spikes", 2: "voltage"}
            os.makedirs("./result_random_nav/", exist_ok=True)
            for num in plt.get_fignums():
                plt.figure(num)
                plt.savefig("./result_random_nav/"+name_figs[num]+str(idx)+".png")