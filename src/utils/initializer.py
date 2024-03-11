from src.agent import Agent


def get_players_initialized(params):
    players = []
    for i in range(params["env"]["n_players"]):
        players.append(Agent(i, params))
    return players
