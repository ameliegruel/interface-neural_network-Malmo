from AntWorld import World, Navigation, SensoryInformation, AutonomousAgent
from AntWorld import *


# Parameters

ArenaSide = 1000 # the agent's size is 1 block = 1 cm (for an ant) => the agent evolves in an arena of diameter 10m*10m
ArenaFloor = 2
xPos = 0
zPos = 0

video_height = 600
video_width = 800

timeRandomNav = 100 # time during which agent is randomly walking around, leaving pheromones behind him (at the end of that, turn around and try to follow the pheromones) 

nb_world_ticks = 0
plots = {}
last_com = "z"

path_StartToEnd = True
path_EndToStart = False





# RUNNING MISSION

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

world = World(ArenaSide=ArenaSide, ArenaFloor=ArenaFloor, xPos=xPos, zPos=zPos, video_height=video_height, video_width=video_width, timeRandomNav=timeRandomNav)
ant_navigator = Navigation(agent_host=agent_host, ArenaFloor=ArenaFloor)
sensory_info = SensoryInformation()


# Network initialization
print("Network initialization", end=' ')
autonomous_ant = AutonomousAgent()


# Mission initialization
missionXML = world.GetMissionXML()
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
    msg = world_state.observations[-1].text 
    ObsJSON = json.loads(msg)
    ObsEnv = {"xPos": ObsJSON["XPos"], "yPos": ObsJSON["YPos"], "zPos": ObsJSON["ZPos"], "yaw": sensory_info.getYaw(-ObsJSON["Yaw"])}
    ObsEnv["GridPheromonesTrace"] = ObsJSON["PheromonesTrace"] # permet de récupérer ce que l'agent perçoit au niveau des phéromones
    ObsEnv["GridObstacles"] = ObsJSON["Obstacles"] # permet de dire si l'agent s'approche d'un obstacle

    
    direction = ["q","d"]
    if nb_world_ticks < timeRandomNav:
        ant_navigator.addPheromones(ObsEnv,agent_host) 
        com= ant_navigator.randomNavigator(ObsEnv, direction,last_com)
        if nb_world_ticks % 20 == 0:
            ### get ant's vision
            ant_view = np.array([sensory_info.getAntView(video_height,video_width,world_state.video_frames[0].pixels)])
            autonomous_ant.reactionToRandomNavigation(ant_view, phase="learning")

    elif nb_world_ticks == timeRandomNav:
        ant_navigator.addPheromones(ObsEnv,agent_host,turn=True)
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
            com = ant_navigator.followPheromonesPath(ObsEnv, agent_host)

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
                ant_navigator.AgentTurn(turn_to_test_views[direction])  # turn once to the right, then turn twice to the left, each time to get ant's vision
                ant_view = np.array([sensory_info.getAntView(video_height,video_width,world_state.video_frames[0].pixels)])
                autonomous_ant.reactionToRandomNavigation(ant_view, phase="test")
                
                if most_familiar_view["EN"] == None or most_familiar_view["EN"] > autonomous_ant.nb_spikes_EN:
                    most_familiar_view["EN"] = autonomous_ant.nb_spikes_EN
                    most_familiar_view["direction"] = direction
            
            ant_navigator.AgentTurn("q") # agent returns in original axe
            turn_to_most_familar_view = {"right": "q", "ahead": "z", "left": "d"} 
            com = turn_to_most_familar_view[most_familiar_view["direction"]]

        elif path_end == True :
            agent_host.sendCommand("move 0")
            agent_host.sendCommand("turn -1")
            time.sleep(1)
            agent_host.sendCommand("turn 0")
        

    # turn 
    ant_navigator.AgentTurn(com)

    agent_host.sendCommand("move 1")
    time.sleep(0.25)
    agent_host.sendCommand("move 0")

    nb_world_ticks += 1
    last_com = com

print()
print("Mission ended")
# Mission has ended.