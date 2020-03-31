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

if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)

ArenaSide = 1000
ArenaFloor = 2
xPos = 0
zPos = 0

video_height = 300
video_width = 400

# Create random objects in the arena

def GenRandomObject(nb, xPos, zPos, ArenaSide, ArenaFloor):
    RandomObjectXML = ""
    BlockType = "redstone_block"
    ObjHeight = [3,4]
    ObjWidth = [1,2]
    PreviousX = [xPos]
    PreviousZ = [zPos]
    max = old_div(ArenaSide,2)
    for i in range(nb) :
        height = random.choice(ObjHeight)
        width = random.choice(ObjWidth)
        x = xPos
        z = zPos
        while x in PreviousX :
            x0 = random.randint(-max,max)
        while z in PreviousZ :
            z0 = random.randint(-max,max)
        for [x,y,z] in [[x0+x, ArenaFloor+y, z0+z] for x in range(width) for y in range(height) for z in range(width)]:
            RandomObjectXML += '<DrawBlock type="' + BlockType + '" x="' + str(x) + '" y="' + str(y) + '" z="' + str(z) + '"/>'
            PreviousX.append(x)
            PreviousZ.append(z)
    return RandomObjectXML


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
    # print("left : (a = ",aLeft," ; b = ",bLeft," )")

    # get the right line
    right = yaw - vf/2
    aRight = getA(DistanceMax,right,pos)
    bRight = getB(aRight,right,pos)
    # print("right : (a = ",aRight," ; b = ",bRight," )")

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
    id_RGBD = id_pixels = 0
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
    print("Size ant_view :", len(ant_view))
    print("Video size :", video_width*video_height)
    return ant_view


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
    print(DistanceObjects)
    return DistanceObjects


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
                    <FlatWorldGenerator generatorString="3;1,2*2,9*3,2;2;"/>
                    <DrawingDecorator>
                        <DrawCuboid type="air" x1="''' + str(old_div(-ArenaSide,2)) + '''" y1="''' + str(ArenaFloor) + '''" z1="''' + str(old_div(-ArenaSide,2)) + '''" x2="''' + str(old_div(ArenaSide,2)) + '''" y2="20" z2="''' + str(old_div(ArenaSide,2)) + '''"/>
                        ''' + GenRandomObject(20, xPos, zPos, ArenaSide, ArenaFloor) + '''
                    </DrawingDecorator>
                    <ServerQuitFromTimeUp timeLimitMs="3600000"/>
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
        ObsEnv["FrontEnv"] = GetFrontVision(ObsJSON["FrontEnv21x21"],21,ObsEnv["yaw"])
        ObsEnv["objects"] = GetObjects(ObsEnv['FrontEnv'],21)
        
        # VisualizeFrontVison(ObsEnv["FrontEnv"])  # => to visualize the frontal vision as seen by the agent

        ### the important informations are contained in the dictionnary ObsEnv
        # print(ObsEnv.keys())
        
        # print("")
        # VisualizeFrontVison(ObsEnv["FrontEnv"])

        ant_view = getAntView(video_height,video_width,world_state.video_frames[0].pixels)
        f = open("ant_view.txt",'w')
        for value in ant_view:
            f.write(str(value)+" ")
        f.close()

        # pix_id = 1
        # dico_id = {1: "R: ", 2: "G: ", 3: "B: ", 4: "depth: "}
        # for pix in world_state.video_frames[0].pixels[479980:480020]:
        #     print(dico_id[pix_id], pix)
        #     if pix_id < 4 :
        #         pix_id += 1
        #     else :
        #         pix_id = 1
        
        com=input("commande : ")
        if com=="z":
            agent_host.sendCommand("move 1")
            time.sleep(0.25)
            agent_host.sendCommand("move 0")
        elif com=="q":
            agent_host.sendCommand("turn -1")
            time.sleep(0.25)
            agent_host.sendCommand("turn 0")
        elif com=="d":
            agent_host.sendCommand("turn 1")
            time.sleep(0.25)
            agent_host.sendCommand("turn 0")
        else : 
            com=input("command (only q, z or d): ")

print()
print("Mission ended")
# Mission has ended.