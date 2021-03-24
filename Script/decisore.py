import pandas as pd

BEST_SCORER = 0.2

player_ID = 462             #Cristiano Ronaldo
match_day = 27

# reading execl files
file_path = "D:\\UniDispense\\ICON\\ICON_Project\\Dataset\\"
all_players = pd.read_excel(file_path + "Dataset_g26_NOPOR.xlsx", index_col="ID")
calendario = pd.read_excel(file_path + "Calendario_2021.xlsx")
classifica = pd.read_excel(file_path + "Statistiche_g26_v2.0.xlsx", sheet_name="Classifica", index_col="Squadra")

player = all_players.loc[player_ID]

player_team = player["Squadra"]

# retrieving matches for the day
matches = list(calendario[match_day])

# retrieving opponent team for the player specified
for match in matches:
    if player_team in match:
        teams_of_match = match.split("-")

for team in teams_of_match:
    if team != player_team:
        vs_team = team

# player and opponent team in rank (whole row)
p_team = classifica.loc[player_team]                    # Juventus - pos. 3
vs_team = classifica.loc[vs_team]                       # Cagliari - pos. 17

# deviation between each team position
dev_pos = vs_team["Pos"] - p_team["Pos"]                # FIRST METRIC. Right value for id = 462 is 14

# deviation between goals scored and conceded
vs_dev_goals = vs_team["Diff_reti"]*(-1)                # SECOND METRIC. Right value for id = 462 is 14

p_dev_goals = p_team["Diff_reti"]                       # THIRD METRIC. Right value for id = 462 is 30

# bonus applied in case the player is the best scorer of his team
bonus_best_scorer = 0                                   # By default 0, if player is not the best scorer of the team
name_player = player["Nome_Cognome"].split("\\")

if name_player[0] in p_team["Miglior_marcatore"]:
    bonus_best_scorer = BEST_SCORER                     # FOURTH METRIC. Right value for id = 462 is 0.2

last_five = p_team["Ultime_5"]                          # last five results for the player team
lf_ratio_p_team = last_five.count("V")/5                # FIFTH METRIC. Right value for id = 462 is 3/5 = 0.6

last_five = vs_team["Ultime_5"]                         # last five results for the opponent team
lf_ratio_vs_team = (-1)*(last_five.count("V")/5)        # SIXTH METRIC. Right value for id = 462 is -(2/5) = -0.4

final_weight = round((dev_pos + vs_dev_goals + p_dev_goals + bonus_best_scorer + lf_ratio_p_team + lf_ratio_vs_team)/100, 3)

print(final_weight)                                     # final weight for the single player. Right value for id = 462 is 0.584
