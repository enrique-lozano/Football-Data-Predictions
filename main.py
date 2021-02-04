import csv
import os
import pandas as pd

class Team:       
    def __init__(self, name:str):           
        self.name = name           
        self.matches = 0
        self.home_matches = 0
        self.away_matches = 0
        self.home_goals = 0
        self.home_goals_against = 0
        self.away_goals = 0
        self.away_goals_against = 0
        self.home_shots = 0
        self.home_shots_against = 0
        self.away_shots = 0
        self.away_shots_against = 0
        self.home_shotsT = 0
        self.home_shotsT_against = 0
        self.away_shotsT = 0
        self.away_shotsT_against = 0
        self.home_corners = 0
        self.home_corners_against = 0
        self.away_corners = 0
        self.away_corners_against = 0
        self.home_faults = 0
        self.home_faults_against = 0
        self.away_faults = 0
        self.away_faults_against = 0
        self.home_yellows = 0
        self.home_yellows_against = 0
        self.away_yellows = 0
        self.away_yellows_against = 0
        self.home_reds = 0
        self.home_reds_against = 0
        self.away_reds = 0
        self.away_reds_against = 0
        

def main():
    print ("WELCOME! This is my football data analizer.")
    print ("--------------------")
    print ("This code has been fully developed by enriqueloz88. Despite being an open source project, its use for commercial purposes without my express authorization is not allowed.")
    print ("--------------------\n")

    df = pd.read_csv('db/2020/E0.csv')

    teams = df['HomeTeam'].unique()
    teams.sort()

    teamsObj = []

    for team in teams:
        t = Team(team)

        for x in range(len(df.index)):
            if team==df['HomeTeam'].values[x]:
                t.matches = t.matches + 1
                t.home_matches = t.home_matches + 1

                t.home_goals = t.home_goals + df['FTHG'].values[x]
                t.home_goals_against = t.home_goals_against + df['FTAG'].values[x]
                t.home_shots = t.home_shots + df[''].values[x]
                t.home_shots_against = t.home_shots_against + df[''].values[x]         
                t.home_shotsT = t.home_shotsT + df[''].values[x]
                t.home_shotsT_against = t.home_shotsT_against + df[''].values[x]             
                t.home_corners = t.home_corners + df[''].values[x]
                t.home_corners_against = t.home_corners_against + df[''].values[x]              
                t.home_faults = t.home_faults + df[''].values[x]
                t.home_faults_against = t.home_faults_against + df[''].values[x]              
                t.home_yellows = t.home_yellows + df[''].values[x]
                t.home_yellows_against = t.home_yellows_against + df[''].values[x]              
                t.home_reds = t.home_reds + df[''].values[x]
                t.home_reds_against = t.home_reds_against + df[''].values[x]
                

            if team==df['AwayTeam'].values[x]:
                t.matches = t.matches + 1
                t.away_matches = t.away_matches + 1
                
                t.away_goals = t.away_goals + df[''].values[x]
                t.away_goals_against = t.away_goals_against + df[''].values[x]
                t.away_shots = t.away_shots + df[''].values[x]
                t.away_shots_against = t.away_shots_against + df[''].values[x]
                t.away_shotsT = t.away_shotsT + df[''].values[x]
                t.away_shotsT_against = t.away_shotsT_against + df[''].values[x]
                t.away_corners = t.away_corners + df[''].values[x]
                t.away_corners_against = t.away_corners_against + df[''].values[x]
                t.away_faults = t.away_faults + df[''].values[x]
                t.away_faults_against = t.away_faults_against + df[''].values[x]
                t.away_yellows = t.away_yellows + df[''].values[x]
                t.away_yellows_against = t.away_yellows_against + df[''].values[x]
                t.away_reds = t.away_reds + df[''].values[x]
                t.away_reds_against = t.away_reds_against + df[''].values[x]

                        


        teamsObj.append(t)

    print(teamsObj[1].home_matches)

main()