import pandas as pd
import seaborn as sns
import numpy as np


score16 = pd.read_csv('./TestScores16.csv')
score16 = score16.drop([ 'Sch Type', 'Low\rGrade', 'High\rGrade', 'Subject'],axis=1)
score16.PassRate = pd.to_numeric(score16.PassRate, errors='coerce')
score16 = score16.groupby(['Div Name','School Name']).mean().reset_index()
score16.columns = ['Division Name', 'School Name', 'PassRate']
#print (list(score16))
#print (score16.head(5))

##Race
ethnicity16 = pd.read_csv('./2016 Ethnicity.csv') 
ethnicity16 = ethnicity16.drop(['Division No.', 'School No.',  'Grade', 'Total\rFull-time Students', 'Part-time Students',  'Hispanic', 'American_Indian_Alaska_Native ', 'Asian', 'Black', 'Native_Hawaiian_Pacific_Islander ','Two_More_Races'],axis=1)
ethnicity16['Totals'] = pd.to_numeric(ethnicity16['Totals'], errors='coerce')
ethnicity16 = ethnicity16.groupby([ 'Division Name',"School Name"]).sum().reset_index()
ethnic_per = ethnicity16.copy()
ethnic_per["White"] = (ethnicity16["White"]/ethnicity16['Totals'])
ethnic_per["Minority"] = (1-ethnic_per["White"])
ethnic_per = ethnic_per.drop("Totals", axis=1)
#print(list(ethnic_per))
#print(ethnic_per.head(5))


##Financial Well Being
ethnicity16 = ethnicity16.drop("White", axis=1)
economic16 = pd.read_csv('./2016 Disadvantaged.csv') 
delete = ['Div. No.', 'School No.','Low Grade', 'High Grade', 'Grade PK', 'Grade JK', 'Grade KG', 'Grade T1', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4', 'Grade 5', 'Grade 6', 'Grade 7', 'Grade 8', 'Grade 9', 'Grade 10', 'Grade 11', 'Grade 12', 'Grade PG', 'Total Full-Time', 'Part-time Students']
economic16 = economic16.drop(delete, axis=1)
economic16.columns = ['Division Name', 'School Name', 'Total Disadvantaged']
economic = pd.merge(ethnicity16, economic16, on=['Division Name', "School Name"])
economic_per = economic
economic_per['Total Disadvantaged'] = (economic['Total Disadvantaged']/economic ['Totals'])
economic_per = economic_per.drop("Totals", axis=1)
#print(list(economic_per))
#print(economic_per.head(5))

#----------Data------------#
data = pd.merge(score16, ethnic_per, on=['Division Name', 'School Name'])
data = pd.merge(data, economic_per,  on=['Division Name', "School Name"])
#print(data.head(5))


