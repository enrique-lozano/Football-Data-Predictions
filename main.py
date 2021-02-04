import csv
import os
import pandas as pd

class Team:   

    def __init__(self, name:str):           
        self.Name = name
        # ---- [H/A]X[F/A] -----
        # H: Home, A: Away | F: For, A: Against
        # ---------------------- 
        self.HM = 0   # Matches
        self.AM = 0
        self.HGF = 0  # Goals
        self.HGA = 0
        self.AGF = 0
        self.AGA = 0
        self.HSF = 0  # Shots
        self.HSA = 0
        self.ASF = 0
        self.ASA = 0
        self.HSTF = 0 # Shots on Target 
        self.HSTA = 0
        self.ASTF = 0
        self.ASTA = 0
        self.HCF = 0  # Corners
        self.HCA = 0
        self.ACF = 0
        self.ACA = 0
        self.HFF = 0  # Faults
        self.HFA = 0
        self.AFF = 0
        self.AFA = 0
        self.HYF = 0  # Yellow cards
        self.HYA = 0
        self.AYF = 0
        self.AYA = 0
        self.HRF = 0  # Red cars
        self.HRA = 0
        self.ARF = 0
        self.ARA = 0    
    
    # Convert to dict. 
    # Return -> {'name': self.name, 'M': self.M,...}
    def as_dict(self):
        return self.__dict__ 


    # Dynamic parameters
    def updateParameters(self):
        self.M = self.HM + self.AM
        self.GF = self.HGF + self.AGF
        self.GA = self.HGA + self.AGA
        self.SF = self.HSF + self.ASF
        self.SA = self.HSA + self.ASA
        self.STF = self.HSTF + self.ASTF
        self.STA = self.HSTA + self.ASTA
        self.CF = self.HCF + self.ACF
        self.CA = self.HCA + self.ACA
        self.FF = self.HFF + self.AFF
        self.FA = self.HFA + self.AFA
        self.YF = self.HYF + self.AYF
        self.YA = self.HYA + self.AYA
        self.RF = self.HRF + self.ARF
        self.RA = self.HRA + self.ARA


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
            t.HM = t.HM + 1
            t.HGF = t.HGF + df['FTHG'].values[x]
            t.HGA = t.HGA + df['FTAG'].values[x]
            t.HSF = t.HSF + df['HS'].values[x]
            t.HSA = t.HSA + df['AS'].values[x]         
            t.HSTF = t.HSTF + df['HST'].values[x]
            t.HSTA = t.HSTA + df['AST'].values[x]             
            t.HCF = t.HCF + df['HC'].values[x]
            t.HCA = t.HCA + df['AC'].values[x]              
            t.HFF = t.HFF + df['HF'].values[x]
            t.HFA = t.HFA + df['AF'].values[x]              
            t.HYF = t.HYF + df['HY'].values[x]
            t.HYA = t.HYA + df['AY'].values[x]              
            t.HRF = t.HRF + df['HR'].values[x]
            t.HRA = t.HRA + df['AR'].values[x]
            t.updateParameters()
            

        if team==df['AwayTeam'].values[x]:
            t.AM = t.AM + 1
            t.AGF = t.AGF + df['FTAG'].values[x]
            t.AGA = t.AGA + df['FTHG'].values[x]
            t.ASF = t.ASF + df['AS'].values[x]
            t.ASA = t.ASA + df['HS'].values[x]
            t.ASTF = t.ASTF + df['AST'].values[x]
            t.ASTA = t.ASTA + df['HST'].values[x]
            t.ACF = t.ACF + df['AC'].values[x]
            t.ACA = t.ACA + df['HC'].values[x]
            t.AFF = t.AFF + df['AF'].values[x]
            t.AFA = t.AFA + df['HF'].values[x]
            t.AYF = t.AYF + df['AY'].values[x]
            t.AYA = t.AYA + df['HY'].values[x]
            t.ARF = t.ARF + df['AR'].values[x]
            t.ARA = t.ARA + df['HR'].values[x]
            t.updateParameters()       

    teamsObj.append(t)

df = pd.DataFrame([x.as_dict() for x in teamsObj])


print(df)

