from __future__ import annotations
import random
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from ray.rllib.env import MultiAgentEnv
from marl.sequential_social_dilemma_games.social_dilemmas.envs.agent import \
    ForrestAgent

MAPS = ["0000000000",
        "0000000000",
        "0000000000",
        "0000000000",
        "0000000000",
        "0000000000",
        "0000000000",
        "0000000000",
        "0000000000",
        "0000000000"]

NearBy = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]

DEFAULT_COLOURS = {' ': [0, 0, 0],  # Black background
                   '0': [0, 0, 0],  # Black background beyond map walls
                   '': [180, 180, 180],  # Grey board walls
                   '@': [180, 180, 180],  # Grey board walls
                   'A': [0, 255, 0],  # Green apples
                   'F': [255, 255, 0],  # Yellow fining beam
                   'P': [159, 67, 255],  # Purple player

                   # Colours for agents. R value is a unique identifier
                   '1': [159, 67, 255],  # Purple
                   '2': [2, 81, 154],  # Blue
                   '3': [204, 0, 204],  # Magenta
                   '4': [216, 30, 54],  # Red
                   '5': [254, 151, 0],  # Orange
                   '6': [100, 255, 255],  # Cyan
                   '7': [99, 99, 255],  # Lavender
                   '8': [250, 204, 255],  # Pink
                   '9': [238, 223, 16]}  # Yellow

tools = {"0": "nothing",  "1":"stn",  "2":"l_stn",  "3":"stk",  "4":"p_stk"}


class ForestEnv(MultiAgentEnv):
    """
    Forest environment
    """

    # MOVES
    # 0: travel  1: shake    2: fold  3: poke  4: bash     5: chase
    # 6: rub     7: scratch  8: peel  9: swap  10: pickup  11: drop

    # Resources
    # 0: Nothing  1: tree    2: berries 3: nettles  4: termite nest
    # 5: coconuts 6: monkeys 7: thorns  8: beehive  9: wasp nest

    # Tools
    # 0: Nothing  1:stones  2: large stones  3: sticks  4: peeled sticks

    def __init__(self, res_map, tool_map, num_agents=1):
        """
        Initialise the environment.
        """

        # Env confinguration
        self.num_agents = num_agents

        self.action_space = Discrete(12)

        # 9 resources, 3 tools (1 peeled), Presense of other agent, 9 resources in a random nearby location, 8 positions for this resource
        self.observation_space = MultiDiscrete([10, 5, 2, 10, 8])

        self.agents = {}
        self.setup_agents()
        self.res_pos = {}
        self.tool_pos = {}
        for i in range(10):
            for j in range(10):
                if res_map[i][j] != "0":
                    self.res_pos[(i,j)] = res_map[i][j]
                if tool_map[i][j] != "0":
                    self.tool_pos[(i,j)] = tool_map[i][j]
        self.res_map = res_map
        self.tool_map = tool_map
        self.base_res_map = res_map
        self.base_tool_map = tool_map

    def setup_agents(self):
        for i in range(self.num_agents):
            agent_id = str(i)
            spawn_point = [random.randint(0, 9), random.randint(0,9)]
            agent = ForrestAgent(agent_id, spawn_point)
            self.agents[agent_id] = agent

    def reset(self, seed: int | None = None, options: dict = dict()) -> tuple:
        """
        Reset the environment.

        Parameters
        ----------
        seed: int | None
            Random seed.
        options: dict
            Optional dict that allows you to define the start (`start_loc` key) and goal (`goal_loc`key) position when resetting the env. By default options={}, i.e. no preference is expressed for the start and goal states and they are randomly sampled.
        """

        # Set seed
        super().reset(seed=seed)

        self.agents = {}
        self.setup_agents()
        for i in range(10):
            for j in range(10):
                if self.base_res_map[i][j] != "0":
                    self.res_pos[(i,j)] = self.base_res_map[i][j]
                if self.base_tool_map[i][j] != "0":
                    self.tool_pos[(i,j)] = self.base_tool_map[i][j]
        self.res_map = self.base_res_map
        self.tool_map = self.base_tool_map

        observations = {}
        for agent in self.agents.values():
            local_res = self.res_map[agent.pos[0]][agent.pos[1]]
            local_tool = self.tool_map[agent.pos[0]][agent.pos[1]]
            other_agent_present = 0
            for other_agent in self.agents.values():
                if other_agent.agent_id != agent.agent_id and other_agent.pos == agent.pos:
                    other_agent_present = 1
                    break
            valid_nearby = self.get_valid_nearby(agent)
            nearby_pos = random.randint(0,len(valid_nearby)-1)
            res_nearby = self.res_map[valid_nearby[nearby_pos][0]][valid_nearby[nearby_pos][1]]
            observations[agent.agent_id] = np.array([int(local_res), int(local_tool), int(other_agent_present), int(res_nearby), int(nearby_pos)])
        return observations, {}

    def get_valid_nearby(self, agent):
        x, y = agent.pos[0], agent.pos[1]
        ret = []
        for x_diff, y_diff in NearBy:
            if 0 <= x+x_diff < 10 and 0 <= y+y_diff < 10:
                ret.append((x + x_diff, y + y_diff))
        return ret

    def step(self, actions):
        """
        Take a step in the environment.
        """
        obs = None
        reward = None
        done = False
        return obs, reward, done, False, None


    def render(self):
        """
        Render the environment.
        """
        labels = [[""]*10 for i in range(10)]
        for agent in self.agents.values():
            if agent.holding == "0":
                labels[agent.pos[0]][agent.pos[1]] += "{} ".format(agent.agent_id)
            else:
                labels[agent.pos[0]][agent.pos[1]] += "{}({}) ".format(agent.agent_id, tools[agent.holding])

        for tk in self.tool_pos.keys():
            if labels[tk[0]][tk[1]] != "":
                labels[tk[0]][tk[1]] += "\n"
            labels[tk[0]][tk[1]] += tools[self.tool_pos[tk]]

        res_cmap = [list(s) for s in self.res_map]
        for i in range(10):
            for j in range(10):
                res_cmap[i][j] = int(res_cmap[i][j])
        cmap = ListedColormap(['white', 'green', 'xkcd:chartreuse', 'tab:brown',
                               'xkcd:sienna', 'red', 'grey', 'yellow', 'tab:orange'])
        sns.heatmap(res_cmap, cmap=cmap, annot=labels, fmt='s', cbar=False, linewidths=1, linecolor='black')
        sns.set(font_scale=0.5)
        plt.show()
        return labels
