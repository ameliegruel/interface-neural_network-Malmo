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
from skimage.transform import resize
import numpy as np
from PIL import Image

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

timeRandomNav = 500 # time during which agent is randomly walking around, leaving pheromones behind him (at the end of that, turn around and try to follow the pheromones) 

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
                    <ServerQuitFromTimeUp timeLimitMs="3600000"/>
                    <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>
              
              <AgentSection mode="Survival">
                <Name>MalmoTutorialBot</Name>
                <AgentStart>
                    <Placement x="'''+ str(xPos) +'''" y="''' + str(ArenaFloor) + '''" z="'''+ str(zPos) + '''" yaw="0"/>
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
                        <Grid name="PheromonesTrace">
                            <min x="-1" y="-1" z="-1"/>
                            <max x="1" y="-1" z="1"/>
                        </Grid>
                    </ObservationFromGrid>
                    <AgentQuitFromTouchingBlockType>
                        <Block type="diamond_block"/>
                    </AgentQuitFromTouchingBlockType>
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
    
    # resize in image of 10*39 pixels (see Ardin et al)
    ant_view = np.array(ant_view)
    ant_view.shape = (-1, w)
    ant_view = resize(ant_view, (10,36))

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

def getWhatIsInFront(agentYaw):
    whatIsInFront = {
        # range(-180, -157): [0,1,2], # 1/2 of north square => centered on index=1 of PheromonesTrace grid
        # range(-157, -112): [3,0,1], # north-west square => centered on index=0 of PheromonesTrace grid
        # range(-112, -67): [6,3,0], # west block => centered on index=3 of PheromonesTrace grid
        # range(-67, -22): [7,6,3], # south-west square => centered on index=6 of PheromonesTrace grid        
        # range(-22, 22): [8,7,6], # south square => centered on index=7 of PheromonesTrace grid
        # range(22, 67): [5,8,7], # south-est square => centered on index=8 of PheromonesTrace grid
        # range(67, 112): [2,5,8], # est square => centered on index=5 of PheromonesTrace grid
        # range(112, 157): [1,2,5], # north-est square => centered on index=2 of PheromonesTrace grid
        # range(157, 181): [0,1,2] # 1/2 of north square => centered on index=1 of PheromonesTrace grid
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
    for yaw in whatIsInFront.keys():
        if agentYaw in yaw:
            print("YESSSSS", yaw, "=>", whatIsInFront[yaw])
            return whatIsInFront[yaw]

def followPheromonesPath(ObsEnv,agent):
    print(ObsJSON["PheromonesTrace"], ObsEnv["yaw"])
    BlocksInFront = getWhatIsInFront(round(ObsEnv["yaw"]))
    print(BlocksInFront)
    directions = {
        # BlocksInFront[0]: "q",
        # BlocksInFront[1]: "z",
        # BlocksInFront[2]: "d"
        BlocksInFront[0]: "q",
        BlocksInFront[1]: "q",
        BlocksInFront[2]: "z",
        BlocksInFront[3]: "d",
        BlocksInFront[4]: "d"
    }
    BlocksWithGold = list(filter(lambda x: ObsEnv["PheromonesTrace"][x] == 'gold_block', BlocksInFront))
    print(BlocksWithGold)
    if BlocksWithGold == [] :
        return random.choice(["q","d"])
    elif BlocksInFront[2] in BlocksWithGold:
        return directions[BlocksInFront[2]]
    else:
        return directions[random.choice(BlocksWithGold)]










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

nb_world_ticks = 1
last_com = "z"

# Loop until mission ends:
while world_state.is_mission_running :
    print(".", end="")
    time.sleep(0.1)
    world_state = agent_host.getWorldState()
    for error in world_state.errors:
        print("Error:",error.text)
    
    # Observations
    if world_state.number_of_observations_since_last_state > 0 :
        msg = world_state.observations[-1].text 
        ObsJSON = json.loads(msg)
        ObsEnv = {"xPos": ObsJSON["XPos"], "yPos": ObsJSON["YPos"], "zPos": ObsJSON["ZPos"], "yaw": getYaw(-ObsJSON["Yaw"])}
        # ObsEnv["FrontEnv"] = GetFrontVision(ObsJSON["FrontEnv21x21"],21,ObsEnv["yaw"])
        # ObsEnv["objects"] = GetObjects(ObsEnv['FrontEnv'],21)
        ObsEnv["PheromonesTrace"] = ObsJSON["PheromonesTrace"]

        ### get ant's vision
        # ant_view = getAntView(video_height,video_width,world_state.video_frames[0].pixels)
        # visualizeAntVision(ant_view,nb_world_ticks)


        movements = ["z","q","d"]
        if nb_world_ticks <= timeRandomNav:
            addPheromones(ObsEnv,agent_host) 
            com=randomNavigator(movements,last_com)
            # move for 1 world tick
            agent_host.sendCommand("move 1")
        elif nb_world_ticks == timeRandomNav+1:
            addPheromones(ObsEnv,agent_host,turn=True)
            print("Follow the pheromones path")
            # add a column inside the arena at (x,y,z) original of the agent : when it reaches the colum, it has reached its point of departure and the mission is ended
            for height in range(ArenaFloor, ArenaFloor+4):
                agent_host.sendCommand('chat /setblock ' + str(xPos) + ' ' + str(height) + ' ' + str(zPos) + ' diamond_block')
            # faces the other way
            agent_host.sendCommand("move 0")
            agent_host.sendCommand("turn -1")
            time.sleep(1)
            agent_host.sendCommand("turn 0")
        else : 
            com = followPheromonesPath(ObsEnv, agent_host)

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
        
        nb_world_ticks += 1
        last_com = com

print()
print("Mission ended")
# Mission has ended.