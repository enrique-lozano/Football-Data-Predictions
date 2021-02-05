import pandas as pd
from tkinter import *


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

data = pd.read_csv('db/2020/E0.csv')
#data = pd.read_csv('input2.csv')
#data3 = pd.read_csv('input3.csv')
#data = data.merge(data2, how='outer')
#data = data.merge(data3, how='outer')
original_data = data

eliminate = [0,1,2] #eliminate division, date and time
for x in range(24, len(data.columns)):
	eliminate.append(x)

#data.drop(['B365CA', 'B365CH', 'B365CD', '', '', '', ''], axis=1)
data = data.drop(data.columns[eliminate], axis=1)  # data.columns is zero-based pd.Index 

#New matches added to predict
'''
new_row = []
for i in range(len(data.columns)):
	new_row.append(0)
new_row[0] = "Barcelona"
new_row[1] = "Real Madrid"
data.loc[len(data)] = new_row
'''

#Partidos
data['P1'] = 0
data['P2'] = 0
data['HM'] = 0
data['AM'] = 0

# A favor actuando como local
# En contra actuando como local
# A favor actuando como visitante
# En contra actuando como visitante
# Total (local+visitante) a favor del equipo que actua en el partido actual como local
# Total (local+visitante) en contra del equipo que actua en el partido actual como local
# Total (local+visitante) a favor del equipo que actua en el partido actual como visitante
# Total (local+visitante) en contra del equipo que actua en el partido actual como visitante

# New columns to add to the dataframe. Will have the total stadistics off all teams until
# the day of the match of the row. In the first match will be 0

newAtr = ['THG','THGA','TAG','TAGA','TG1','TGA1','TG2','TGA2','THS','THSA','TAS',
'TASA','TS1','TSA1','TS2','TSA2','THST','THSTA','TAST','TASTA','TST1','TSTA1','TST2',
'TSTA2','THC','THCA','TAC','TACA','TC1','TCA1','TC2','TCA2','THF','THFA','TAF','TAFA',
'TF1','TFA1','TF2','TFA2','THY','THYA','TAY','TAYA','TY1','TYA1','TY2','TYA2','THR',
'THRA','TAR','TARA','TR1','TRA1','TR2','TRA2']

for atr in newAtr:
	data[atr] = 0

#Other ratings
data['ELOG1'] = 0.0 #ELO rating in goals. data = data + (actual-medium)
data['ELOG2'] = 0.0
data['LG1'] = 0.0   #Last games results. data=data*0.5+POINTS     
data['LG2'] = 0.0   #Last games results. data=data*0.5+POINTS   

teams = pd.unique(data['HomeTeam'])
teams.sort()

teamsList = []

for team in teams:
	matches = 0
	home_matches = 0
	away_matches = 0
	home_goals = 0
	home_goals_against = 0
	away_goals = 0
	away_goals_against = 0
	home_shots = 0
	home_shots_against = 0
	away_shots = 0
	away_shots_against = 0
	home_shotsT = 0
	home_shotsT_against = 0
	away_shotsT = 0
	away_shotsT_against = 0
	home_corners = 0
	home_corners_against = 0
	away_corners = 0
	away_corners_against = 0
	home_faults = 0
	home_faults_against = 0
	away_faults = 0
	away_faults_against = 0
	home_yellows = 0
	home_yellows_against = 0
	away_yellows = 0
	away_yellows_against = 0
	home_reds = 0
	home_reds_against = 0
	away_reds = 0
	away_reds_against = 0

	t = Team(team) # Create new object team
	for x in range(len(data.index)):
		if team==data['HomeTeam'].values[x]:
			data['P1'].values[x] = matches
			data['HM'].values[x] = home_matches
			home_matches = home_matches + 1
			matches = matches + 1
			data['THG'].values[x] = home_goals
			data['THGA'].values[x] = home_goals_against
			data['TG1'].values[x] = home_goals + away_goals
			data['TGA1'].values[x] = home_goals_against + away_goals_against
			home_goals = home_goals + data['FTHG'].values[x]
			home_goals_against = home_goals_against + data['FTAG'].values[x]
			data['THS'].values[x] = home_shots
			data['THSA'].values[x] = home_shots_against
			data['TS1'].values[x] = home_shots + away_shots
			data['TSA1'].values[x] = home_shots_against + away_shots_against
			home_shots = home_shots + data['HS'].values[x]
			home_shots_against = home_shots_against + data['AS'].values[x]
			data['THST'].values[x] = home_shotsT
			data['THSTA'].values[x] = home_shotsT_against
			data['TST1'].values[x] = home_shotsT + away_shotsT
			data['TSTA1'].values[x] = home_shotsT_against + away_shotsT_against
			home_shotsT = home_shotsT + data['HST'].values[x]
			home_shotsT_against = home_shotsT_against + data['AST'].values[x]
			data['THC'].values[x] = home_corners
			data['THCA'].values[x] = home_corners_against
			data['TC1'].values[x] = home_corners + away_corners
			data['TCA1'].values[x] = home_corners_against + away_corners_against
			home_corners = home_corners + data['HC'].values[x]
			home_corners_against = home_corners_against + data['AC'].values[x]
			data['THF'].values[x] = home_faults
			data['THFA'].values[x] = home_faults_against
			data['TF1'].values[x] = home_faults + away_faults
			data['TFA1'].values[x] = home_faults_against + away_faults_against
			home_faults = home_faults + data['HF'].values[x]
			home_faults_against = home_faults_against + data['AF'].values[x]
			data['THY'].values[x] = home_yellows
			data['THYA'].values[x] = home_yellows_against
			data['TY1'].values[x] = home_yellows + away_yellows
			data['TYA1'].values[x] = home_yellows_against + away_yellows_against
			home_yellows = home_yellows + data['HY'].values[x]
			home_yellows_against = home_yellows_against + data['AY'].values[x]
			data['THR'].values[x] = home_reds
			data['THRA'].values[x] = home_reds_against
			data['TR1'].values[x] = home_reds + away_reds
			data['TRA1'].values[x] = home_reds_against + away_reds_against
			home_reds = home_reds + data['HR'].values[x]
			home_reds_against = home_reds_against + data['AR'].values[x]

			t.HM = t.HM + 1
			t.HGF = t.HGF + data['FTHG'].values[x]
			t.HGA = t.HGA + data['FTAG'].values[x]
			t.HSF = t.HSF + data['HS'].values[x]
			t.HSA = t.HSA + data['AS'].values[x]         
			t.HSTF = t.HSTF + data['HST'].values[x]
			t.HSTA = t.HSTA + data['AST'].values[x]             
			t.HCF = t.HCF + data['HC'].values[x]
			t.HCA = t.HCA + data['AC'].values[x]              
			t.HFF = t.HFF + data['HF'].values[x]
			t.HFA = t.HFA + data['AF'].values[x]              
			t.HYF = t.HYF + data['HY'].values[x]
			t.HYA = t.HYA + data['AY'].values[x]              
			t.HRF = t.HRF + data['HR'].values[x]
			t.HRA = t.HRA + data['AR'].values[x]
			t.updateParameters()

		if team==data['AwayTeam'].values[x]:
			data['P2'].values[x] = matches
			data['AM'].values[x] = away_matches
			away_matches = away_matches + 1
			matches = matches + 1
			data['TAG'].values[x] = away_goals
			data['TAGA'].values[x] = away_goals_against
			data['TG2'].values[x] = home_goals + away_goals
			data['TGA2'].values[x] = home_goals_against + away_goals_against
			away_goals = away_goals + data['FTAG'].values[x]
			away_goals_against = away_goals_against + data['FTHG'].values[x]
			data['TAS'].values[x] = away_shots
			data['TASA'].values[x] = away_shots_against
			data['TS2'].values[x] = home_shots + away_shots
			data['TSA2'].values[x] = home_shots_against + away_shots_against
			away_shots = away_shots + data['AS'].values[x]
			away_shots_against = away_shots_against + data['HS'].values[x]
			data['TAST'].values[x] = away_shotsT
			data['TASTA'].values[x] = away_shotsT_against
			data['TST2'].values[x] = home_shotsT + away_shotsT
			data['TSTA2'].values[x] = home_shotsT_against + away_shotsT_against
			away_shotsT = away_shotsT + data['AST'].values[x]
			away_shotsT_against = away_shotsT_against + data['HST'].values[x]
			data['TAC'].values[x] = away_corners
			data['TACA'].values[x] = away_corners_against
			data['TC2'].values[x] = home_corners + away_corners
			data['TCA2'].values[x] = home_corners_against + away_corners_against
			away_corners = away_corners + data['AC'].values[x]
			away_corners_against = away_corners_against + data['HC'].values[x]
			data['TAF'].values[x] = away_faults
			data['TAFA'].values[x] = away_faults_against
			data['TF2'].values[x] = home_faults + away_faults
			data['TFA2'].values[x] = home_faults_against + away_faults_against
			away_faults = away_faults + data['AF'].values[x]
			away_faults_against = away_faults_against + data['HF'].values[x]
			data['TAY'].values[x] = away_yellows
			data['TAYA'].values[x] = away_yellows_against
			data['TY2'].values[x] = home_yellows + away_yellows
			data['TYA2'].values[x] = home_yellows_against + away_yellows_against
			away_yellows = away_yellows + data['AY'].values[x]
			away_yellows_against = away_yellows_against + data['HY'].values[x]
			data['TAR'].values[x] = away_reds
			data['TARA'].values[x] = away_reds_against
			data['TR2'].values[x] = home_reds + away_reds
			data['TRA2'].values[x] = home_reds_against + away_reds_against
			away_reds = away_reds + data['AR'].values[x]
			away_reds_against = away_reds_against + data['HR'].values[x]		
			
			t.AM = t.AM + 1
			t.AGF = t.AGF + data['FTAG'].values[x]
			t.AGA = t.AGA + data['FTHG'].values[x]
			t.ASF = t.ASF + data['AS'].values[x]
			t.ASA = t.ASA + data['HS'].values[x]
			t.ASTF = t.ASTF + data['AST'].values[x]
			t.ASTA = t.ASTA + data['HST'].values[x]
			t.ACF = t.ACF + data['AC'].values[x]
			t.ACA = t.ACA + data['HC'].values[x]
			t.AFF = t.AFF + data['AF'].values[x]
			t.AFA = t.AFA + data['HF'].values[x]
			t.AYF = t.AYF + data['AY'].values[x]
			t.AYA = t.AYA + data['HY'].values[x]
			t.ARF = t.ARF + data['AR'].values[x]
			t.ARA = t.ARA + data['HR'].values[x]
			t.updateParameters()  
	
	teamsList.append(t)

''' #TeamList to df
df = pd.DataFrame([x.as_dict() for x in teamsList])
print(df)
'''

print ("\n---------------------------------------------------------")
print ("Team")
print ("Goles anotados en casa:" + str(data['THG'].values[len(data)-1]))

#print(tabulate(data, headers='keys', tablefmt='fancy_grid'))
writer = pd.ExcelWriter("output.xlsx", engine='xlsxwriter')
data.to_excel(writer, index = False, header=True, sheet_name='Sheet1')  
data.to_csv("output.csv", index = False, header=True)  

workbook  = writer.book
worksheet = writer.sheets['Sheet1']
worksheet.set_column(2, 80, 5)
worksheet.set_column(0, 1, 12)
writer.save()

#Coger las medias
for atr in newAtr:
	if atr[0]=='T' and atr[1]=='H':
		data[atr] = data[atr]/data['HM']
	if atr[0]=='T' and atr[1]=='A':
		data[atr] = data[atr]/data['AM']
	if atr[-1]=='1':
		data[atr] = data[atr]/data['P1']
	if atr[-1]=='2':
		data[atr] = data[atr]/data['P2']


data = data.fillna(0)

for team in teams:
	elog = 1.0
	LG = 0.0
	for x in range(len(data.index)):
		if team==data['HomeTeam'].values[x]:
			points = 1
			if data['FTR'].values[x] == 'A':
				points = -3
			if data['FTR'].values[x] == 'H':
				points = 3
			data['LG1'].values[x] =  LG
			LG = LG*0.8 + points

			data['ELOG1'].values[x] = elog
			if data['HM'].values[x]!=0 and data['P1'].values[x]!=0:
				elog = elog + 0.7*(data['FTHG'].values[x] - (data['THG'].values[x]+data['TG1'].values[x])/2)
		if team==data['AwayTeam'].values[x]:
			points = 1
			if data['FTR'].values[x] == 'H':
				points = -3
			if data['FTR'].values[x] == 'A':
				points = 3
			data['LG2'].values[x] = LG
			LG = LG*0.8 + points

			data['ELOG2'].values[x] = elog
			if data['AM'].values[x]!=0 and data['P2'].values[x]!=0:
				elog = elog + 0.7*(data['FTAG'].values[x] - (data['TAG'].values[x]+data['TG2'].values[x])/2)

writer = pd.ExcelWriter("outputMedias.xlsx", engine='xlsxwriter')
data.to_excel(writer, index = False, header=True, sheet_name='Sheet1')  

workbook  = writer.book
worksheet = writer.sheets['Sheet1']
worksheet.set_column(2, 80, 5)
worksheet.set_column(0, 1, 12)
writer.save()

'''---------------------------------------------
--------------MAKING THE MODEL------------------
------------------------------------------------'''

#We eliminate the first games as they are not very representative for the training
data = data.drop(range(0,91), axis=0) 
for x in range(len(data.index)):
	if data['HM'].values[x]<3 or data['P1'].values[x]<4 or data['AM'].values[x]<3 or data['P2'].values[x]<4:
		data.drop(x, axis=0)

#Variable to predict. Dependent variable
Y = data['FTHG'].values
Y = Y.astype('int')
Y2 = data['FTAG'].values
Y2 = Y2.astype('int')

#Indepiendent variables
X = data[['THG','TAG','THGA','TAGA','TG1','TG2','TGA1','TGA2','THS','TAS','THSA','TASA','TS1','TS2','TSA1','TSA2',
'THST','TAST','THSTA','TASTA','TST1','TST2','TSTA1','TSTA2','THC','TAC','THCA','TACA','TC1','TC2','TCA1','TCA2',
'THF','TAF','THFA','TAFA','TF1','TF2','TFA1','TFA2','THY','TAY','THYA','TAYA','TY1','TY2','TYA1','TYA2',
'THR','TAR','THRA','TARA','TR1','TR2','TRA1','TRA2','ELOG1','ELOG2','LG1','LG2']]
#X = data[['THG','TAG','THGA','TAGA','TG1','TG2','TGA1','TGA2','ELOG1', 'ELOG2','LG1','LG2']]
X2 = X
print("\n ---------------------------------")
print("Making random forest.............")

#Split data into train and test datasets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.06, random_state=20)
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2, test_size=0.06, random_state=20)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=180, min_samples_leaf=2, min_samples_split=3, random_state=20)

model.fit(X_train, Y_train)
prediction_test = model.predict(X_test) #Results of the predictions in a list[]

model.fit(X2_train, Y2_train)
prediction_test2 = model.predict(X2_test)

from sklearn import metrics
print ("Mean squared error H= ", '{:.2f}'.format(100*round(metrics.mean_squared_error(Y_test, prediction_test),2)), "%")
print ("Mean absolute error H= ", '{:.2f}'.format(100*round(metrics.mean_absolute_error(Y_test, prediction_test),2)), "%")
print ("Mean squared error A= ", '{:.2f}'.format(100*round(metrics.mean_squared_error(Y2_test, prediction_test2),2)), "%")
print ("Mean absolute error A= ", '{:.2f}'.format(100*round(metrics.mean_absolute_error(Y2_test, prediction_test2),2)), "%")

#Creating new dataframe to print the predictions
matches_prediction = []
i=0
for row in X_test.index:
	match_data = []
	match_data.append(original_data['HomeTeam'].values[row])
	match_data.append(original_data['AwayTeam'].values[row])
	match_data.append(prediction_test[i])
	match_data.append(prediction_test2[i])
	match_data.append(Y_test[i])
	match_data.append(Y2_test[i])
	match_data = tuple(match_data)
	matches_prediction.append(match_data)
	i=i+1

df_prediction = pd.DataFrame(matches_prediction, columns=['HomeTeam', 'AwayTeam', 'PHG', 'PAG','RHG', 'RAG'])
print (df_prediction)

#print best attributes
print("\nBest Attributes:")
feature_list = list(X.columns)
features_imp = pd.Series(model.feature_importances_, index=feature_list).sort_values(ascending=False)
print(features_imp.head())
print("\nWorst Attributes:")
print(features_imp.tail())

print("----------------------------------------")
print("----------------ENDING------------------")
print("----------------------------------------")

#data = data[:-10]
