from __future__ import annotations
import random
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from ray.rllib.env import MultiAgentEnv
from marl.sequential_social_dilemma_games.social_dilemmas.envs.agent import \
    ForestAgent


NearBy = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]

tools = {"0": "nothing",  "1":"stn",  "2":"l_stn",  "3":"stk",  "4":"p_stk"}

res_live = {"0": np.infty, "1": np.infty, "2": 20, "3": 20, "4": 10, "5": 5,
            "6": 2, "7": np.infty, "8": 1, "9": np.infty}

tools_live = {"0": np.infty, "1": np.infty, "2": np.infty, "3": 1000, "4": 500}


class ForestEnv(MultiAgentEnv):
    """
    Forest environment
    """

    # MOVES
    # 0: travel  1: shake    2: fold  3: poke  4: bash     5: chase
    # 6: rub     7: scratch  8: peel  9: swap  10: throw  11: drop

    # Resources
    # 0: Nothing  1: tree    2: berries 3: nettles  4: termite nest
    # 5: coconuts 6: monkeys 7: thorns  8: beehive  9: wasp nest

    # Tools
    # 0: Nothing  1:stones  2: large stones  3: sticks  4: peeled sticks

    def __init__(self, res_map, tool_map, num_agents=250):
        """
        Initialise the environment.
        """

        # Env confinguration
        self.num_agents = num_agents
        self.timestamp = 0

        self.action_space = Discrete(12)

        # 9 resources, 3 tools (1 peeled), Presense of other agent, 9 resources in a random nearby location, 8 positions for this resource
        self.observation_space = MultiDiscrete([10, 5, 2, 10, 8])
        self.agents = {}
        self.last_agent = 0  # last spawned agent id
        self.live_agents = {}  # dict of all living agents
        # self.dead_agents = {}  # dict of all dead agents
        self.setup_agents()
        self.res_pos = {}
        self.tool_pos = {}
        for i in range(10):
            for j in range(10):
                if res_map[i][j] != "0":
                    self.res_pos[(i,j)] = [res_map[i][j], res_live[res_map[i][j]]]
                if tool_map[i][j] != "0":
                    self.tool_pos[(i,j)] = [tool_map[i][j], tools_live[tool_map[i][j]]]
        self.res_map = [list(s) for s in res_map]
        self.tool_map = [list(s) for s in tool_map]
        self.base_res_map = res_map
        self.base_tool_map = tool_map

    def setup_agents(self):
        for i in range(self.num_agents):
            agent_id = str(i)
            spawn_point = [random.randint(0, 9), random.randint(0,9)]
            agent = ForestAgent(agent_id, spawn_point)
            self.agents[agent_id] = agent
            if i == 0:
                self.live_agents[agent_id] = agent

    def reset(self, seed: int | None = None, options: dict = dict()) -> tuple:
        """
        Reset the environment.
        """
        # Set seed
        super().reset(seed=seed)
        self.timestamp = 0
        self.agents = {}
        self.last_agent = 0
        self.live_agents = {}
        # self.dead_agents = {}
        self.setup_agents()
        for i in range(10):
            for j in range(10):
                if self.base_res_map[i][j] != "0":
                    self.res_pos[(i,j)] = [self.base_res_map[i][j], res_live[self.base_res_map[i][j]]]
                if self.base_tool_map[i][j] != "0":
                    self.tool_pos[(i,j)] = [self.base_tool_map[i][j], tools_live[self.base_tool_map[i][j]]]
        self.res_map = [list(s) for s in self.base_res_map]
        self.tool_map = [list(s) for s in self.base_tool_map]

        observations = {}
        for agent in self.live_agents.values():
            local_res = str(self.res_map[agent.pos[0]][agent.pos[1]])
            local_tool = str(self.tool_map[agent.pos[0]][agent.pos[1]])
            other_agent_present = 0
            for other_agent in self.live_agents.values():
                if other_agent.agent_id != agent.agent_id and other_agent.pos == agent.pos:
                    other_agent_present = 1
                    break
            valid_nearby = self.get_valid_nearby(agent)
            nearby_pos = random.randint(0,len(valid_nearby)-1)
            pos, index = valid_nearby[nearby_pos]
            res_nearby = str(self.res_map[pos[0]][pos[1]])
            observations[agent.agent_id] = np.array([int(local_res), int(local_tool), int(other_agent_present), int(res_nearby), int(index)])
        return observations, {}

    def get_valid_nearby(self, agent):
        x, y = agent.pos[0], agent.pos[1]
        ret = []
        i = 0
        for x_diff, y_diff in NearBy:
            if 0 <= x+x_diff < 10 and 0 <= y+y_diff < 10:
                ret.append([(x + x_diff, y + y_diff), i])
            i += 1
        return ret

    def step(self, actions):
        """
        Take a step in the environment.
        """
        obs = {}
        rewards = {}
        dones = {}
        self.timestamp += 1
        for agent in self.live_agents.values():
            agent.get_older(1)
            if agent.age >= 10000:
                self.kill_agent(agent.agent_id)
        if self.timestamp % 400 == 0:
            self.last_agent += 1
            self.live_agents[str(self.last_agent)] = self.agents[str(self.last_agent)]
            actions[str(self.last_agent)] = "0"  # fill the new born agent
        for agent_id, action in actions.items():
            if agent_id in self.live_agents:
                r = self.process_action(agent_id, action)
                rewards[agent_id] = r
                # tool decay
                agent = self.live_agents[agent_id]
                agent.holding_live -= 1
                if agent.holding_live == 0:
                    self.respawn_item(agent.pos[0],agent.pos[1],"3",res=False)
                    agent.holding = "0"
                    agent.holding_live = np.infty
        # tool decay for sticks on the ground
        for tk in self.tool_pos.keys():
            if self.tool_pos[tk][0] == "3" or self.tool_pos[tk][0] == "4":
                self.tool_pos[tk][1] -= 1
                if self.tool_pos[tk][1] == 0:
                    self.tool_pos.pop(tk)
                    self.tool_map[tk[0]][tk[1]] = "0"
                    self.respawn_item(0,0,"3", res=False)
        for agent in self.live_agents.values():
            local_res = str(self.res_map[agent.pos[0]][agent.pos[1]])
            local_tool = str(self.tool_map[agent.pos[0]][agent.pos[1]])
            other_agent_present = 0
            for other_agent in self.live_agents.values():
                if other_agent.agent_id != agent.agent_id and other_agent.pos == agent.pos:
                    other_agent_present = 1
                    break
            valid_nearby = self.get_valid_nearby(agent)
            nearby_pos = random.randint(0,len(valid_nearby)-1)
            pos, index = valid_nearby[nearby_pos]
            res_nearby = str(self.res_map[pos[0]][pos[1]])
            obs[agent.agent_id] = np.array([int(local_res), int(local_tool), int(other_agent_present), int(res_nearby), int(index)])
            agent.last_seen_pos = index
        dones["__all__"] = self.timestamp == 100000
        return obs, rewards, dones, {"__all__": False}, {}

    def kill_agent(self, agent_id):
        self.live_agents.pop(agent_id)

    def process_action(self, agent_id, action):
        # MOVES
        # 0: nothing  1: shake    2: fold  3: poke  4: bash     5: chase
        # 6: rub      7: scratch  8: peel  9: swap  10: throw  11: travel

        # Resources
        # 0: Nothing  1: tree    2: berries 3: nettles  4: termite nest
        # 5: coconuts 6: monkeys 7: thorns  8: beehive  9: wasp nest

        # Tools
        # 0: Nothing  1:stones  2: large stones  3: sticks  4: peeled sticks
        # np.random.normal(mu, sigma)
        reward = 0
        agent = self.live_agents[agent_id]
        local_res = self.res_map[agent.pos[0]][agent.pos[1]]
        local_tool = self.tool_map[agent.pos[0]][agent.pos[1]]
        other_agent_present = 0
        for other_agent in self.live_agents.values():
            if other_agent.agent_id != agent.agent_id and other_agent.pos == agent.pos:
                other_agent_present = 1
                break
        if local_res == "1": # tree
            reward += 1
        elif local_res == "2": # berries
            if action == 1: # shake
                reward += np.random.normal(20, 5)
            else:
                reward += np.random.normal(10, 5)
            self.res_pos[(agent.pos[0],agent.pos[1])][1] -= 1
            if self.res_pos[(agent.pos[0],agent.pos[1])][1] == 0:
                self.respawn_item(agent.pos[0],agent.pos[1],"2",res=True)
        elif local_res == "3": # nettles
            if action == 2: # fold
                reward += np.random.normal(30, 5)
                self.res_pos[(agent.pos[0],agent.pos[1])][1] -= 1
                if self.res_pos[(agent.pos[0],agent.pos[1])][1] == 0:
                    self.respawn_item(agent.pos[0],agent.pos[1],"3",res=True)
            else:
                reward += np.random.normal(-10, 5)
        elif local_res == "4": # termite nest
            if action == 3: # poke
                if agent.holding == "4":
                    reward += np.random.normal(100, 20)
                if agent.holding == "3":
                    reward += np.random.normal(50, 10)
            if action == 4: # bash
                if agent.holding == "1" or agent.holding == "2":
                    reward += np.random.normal(35, 10)
            self.res_pos[(agent.pos[0],agent.pos[1])][1] -= 1
            if self.res_pos[(agent.pos[0],agent.pos[1])][1] == 0:
                self.respawn_item(agent.pos[0],agent.pos[1],"4",res=True)
        elif local_res == "5": # coconut
            if action == 4:
                item_hold = agent.holding
                item_ground = local_tool
                p = 0
                if item_hold == "0":
                    p = 0.01
                elif item_hold == "1":
                    if item_ground == "0":
                        p = 0.2
                    elif item_ground == "1":
                        p = 0.6
                    elif item_ground == "2":
                        p = 1
                elif item_hold == "2":
                    if item_ground == "0":
                        p = 0.4
                    elif item_ground == "1":
                        p = 0.6
                    elif item_ground == "2":
                        p = 0.8
                if random.random() < p:
                    reward += np.random.normal(150, 20)
                    self.res_pos[(agent.pos[0],agent.pos[1])][1] -= 1
                    if self.res_pos[(agent.pos[0],agent.pos[1])][1] == 0:
                        self.respawn_item(agent.pos[0],agent.pos[1],"5",res=True)
        elif local_res == "6": # monkey
            if action == 5 and other_agent_present:
                reward += np.random.normal(200, 20)
                self.res_pos[(agent.pos[0],agent.pos[1])][1] -= 1
                if self.res_pos[(agent.pos[0],agent.pos[1])][1] == 0:
                    self.respawn_item(agent.pos[0],agent.pos[1],"6",res=True)
            elif action == 10:
                p = 0
                if agent.holding == "1":
                    p = 0.1
                elif agent.holding == "2" or agent.holding == "3" or agent.holding == "4":
                    p = 0.02
                agent.holding = "0"
                agent.holding_live = np.infty
                if random.random() < p:
                    reward += np.random.normal(200, 20)
                    self.res_pos[(agent.pos[0],agent.pos[1])][1] -= 1
                    if self.res_pos[(agent.pos[0],agent.pos[1])][1] == 0:
                        self.respawn_item(agent.pos[0],agent.pos[1],"6",res=True)
        elif local_res == "7":
            reward += np.random.normal(-20, 5)
        elif local_res == "8":
            if action == 4 and (agent.holding == "1" or agent.holding == "2"):
                reward += np.random.normal(-200, 50)
                if random.random() < 0.5:
                    reward += np.random.normal(500, 50)
                    self.res_pos[(agent.pos[0],agent.pos[1])][1] -= 1
                    if self.res_pos[(agent.pos[0],agent.pos[1])][1] == 0:
                        self.respawn_item(agent.pos[0],agent.pos[1],"8",res=True)
            else:
                if random.random() < 0.2:
                    reward += np.random.normal(-200, 50)
                if action == 6 and (agent.holding == "3" or agent.holding == "4") \
                        and (local_tool == "3" or local_tool == "4"):
                    reward += np.random.normal(500, 50)
                    self.res_pos[(agent.pos[0],agent.pos[1])][1] -= 1
                    if self.res_pos[(agent.pos[0],agent.pos[1])][1] == 0:
                        self.respawn_item(agent.pos[0],agent.pos[1],"8",res=True)
        elif local_res == "9":
            if action == 4 and (agent.holding == "1" or agent.holding == "2"):
                reward += np.random.normal(-200, 50)
            else:
                if random.random() < 0.2:
                    reward += np.random.normal(-200, 50)
        if action == 7:
            if agent.holding == "3" or agent.holding == "4":
                reward += 2
            else:
                reward += 1
        elif action == 8:
            if agent.holding == "3":
                agent.holding = "4"
                agent.holding_live = 500
        elif action == 9:
            item_hold = agent.holding
            item_hold_live = agent.holding_live
            item_ground = local_tool
            if item_ground != "0" and item_hold != "0":
                agent.holding = item_ground
                agent.holding_live = self.tool_pos[(agent.pos[0],agent.pos[1])][1]
                self.tool_map[agent.pos[0]][agent.pos[1]] = item_hold
                self.tool_pos[(agent.pos[0],agent.pos[1])] = [item_hold, item_hold_live]
            elif item_hold != "0":
                agent.holding = "0"
                agent.holding_live = np.infty
                self.tool_map[agent.pos[0]][agent.pos[1]] = item_hold
                self.tool_pos[(agent.pos[0],agent.pos[1])] = [item_hold, item_hold_live]
            elif item_ground != "0":
                agent.holding = item_ground
                agent.holding_live = self.tool_pos[(agent.pos[0],agent.pos[1])][1]
                self.tool_map[agent.pos[0]][agent.pos[1]] = "0"
                self.tool_pos.pop((agent.pos[0],agent.pos[1]))
        elif action == 11:
            if agent.last_seen_pos is not None:
                x_diff, y_diff = NearBy[agent.last_seen_pos]
                agent.pos = [agent.pos[0]+x_diff, agent.pos[1]+y_diff]

        return reward

    def respawn_item(self, old_x, old_y, item_type, res):
        if res:
            self.res_pos.pop((old_x,old_y))
            self.res_map[old_x][old_y] = "0"
            found = False
            while not found:
                x, y = random.randint(0,9), random.randint(0,9)
                if (x, y)!=(old_x, old_y) and self.res_map[x][y] == "0":
                    self.res_map[x][y] = item_type
                    self.res_pos[(x,y)] = [item_type, res_live[item_type]]
                    found = True
        else:
            found = False
            while not found:
                x, y = random.randint(0,9), random.randint(0,9)
                if self.tool_map[x][y] == "0":
                    self.tool_map[x][y] = item_type
                    self.tool_pos[(x,y)] = [item_type, tools_live[item_type]]
                    found = True


    def render(self):
        """
        Render the environment.
        """
        labels = [[""]*10 for i in range(10)]
        for agent in self.live_agents.values():
            if agent.holding == "0":
                labels[agent.pos[0]][agent.pos[1]] += "{} ".format(agent.agent_id)
            else:
                labels[agent.pos[0]][agent.pos[1]] += "{}({}) ".format(agent.agent_id, tools[agent.holding])

        for tk in self.tool_pos.keys():
            if labels[tk[0]][tk[1]] != "":
                labels[tk[0]][tk[1]] += "\n"
            labels[tk[0]][tk[1]] += tools[self.tool_pos[tk][0]]

        res_cmap = [list(s) for s in self.res_map]
        for i in range(10):
            for j in range(10):
                res_cmap[i][j] = int(res_cmap[i][j])
        cmap = ListedColormap(['white', 'green', 'blue', 'xkcd:chartreuse', 'tab:brown',
                               'xkcd:sienna', 'red', 'grey', 'yellow', "xkcd:beige"])
        sns.heatmap(res_cmap, cmap=cmap, annot=labels, fmt='s', cbar=False, linewidths=1, linecolor='black')
        sns.set(font_scale=0.5)
        plt.show()
        return labels
