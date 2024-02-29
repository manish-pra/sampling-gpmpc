from src.agent import Agent


# def get_players_initialized(train, params, grid_V):
#     players = []
#     init_safe = train["Cx_X"]
#     for i in range(params["env"]["n_players"]):
#         # instantiate players object
#         players.append(Agent(i, train["Cx_X"][i], train["Cx_Y"][i], train["Fx_Y"][i],
#                              params, grid_V))
#         # player i will get to know about the samples collected by other agents using communicate function
#         players[i].communicate_constraint(
#             train["Cx_X"][:i]+train["Cx_X"][i+1:], train["Cx_Y"][:i]+train["Cx_Y"][i+1:])
#         players[i].communicate_density(
#             train["Fx_X"][:i]+train["Fx_X"][i+1:], train["Fx_Y"][:i]+train["Fx_Y"][i+1:])
#     return players

def get_players_initialized(params):
    players = []
    for i in range(params["env"]["n_players"]):
        # instantiate players object
        players.append(Agent(i,params))
    return players