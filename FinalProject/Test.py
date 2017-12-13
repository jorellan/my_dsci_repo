import pandas as pd
import seaborn as sns
import numpy as np


score17 = pd.read_csv('./TestScores17.csv')
score17 = score17.drop([ 'Sch Type', 'Low\rGrade', 'High\rGrade', 'Subject'],axis=1)
score17.PassRate = pd.to_numeric(score17.PassRate, errors='coerce')
score17 = score17.groupby(['Div Name','School Name']).mean().reset_index()
score17.columns = ['Division Name', 'School Name', 'PassRate']
#print (list(score17))
#print (score17.head(5))

##Race
ethnicity17 = pd.read_csv('./2017 Ethnicity.csv') 
ethnicity17 = ethnicity17.drop(['Division No.', 'School No.',  'Grade', 'Total\rFull-time Students', 'Part-time Students',  'Hispanic', 'American_Indian_Alaska_Native', 'Asian', 'Black', 'Native_Hawaiian_Pacific_Islander','Two_More_Races'],axis=1)
ethnicity17['Totals'] = pd.to_numeric(ethnicity17['Totals'], errors='coerce')
ethnicity17 = ethnicity17.groupby([ 'Division Name',"School Name"]).sum().reset_index()
ethnic_per = ethnicity17.copy()
ethnic_per["White"] = (ethnicity17["White"]/ethnicity17['Totals'])
ethnic_per["Minority"] = (1-ethnic_per["White"])
ethnic_per = ethnic_per.drop("Totals", axis=1)
#print(list(ethnic_per))
#print(ethnic_per.head(5))


##Financial Well Being
ethnicity17 = ethnicity17.drop("White", axis=1)
economic17 = pd.read_csv('./2017 Disadvantaged.csv') 
delete = ['Div. No.', 'School No.','Low Grade', 'High Grade', 'Grade PK', 'Grade JK', 'Grade KG', 'Grade T1', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4', 'Grade 5', 'Grade 6', 'Grade 7', 'Grade 8', 'Grade 9', 'Grade 10', 'Grade 11', 'Grade 12', 'Grade PG', 'Total Full-Time', 'Part-time Students']
economic17 = economic17.drop(delete, axis=1)
economic17.columns = ['Division Name', 'School Name', 'Total Disadvantaged']
economic17['Total Disadvantaged'] = pd.to_numeric(economic17['Total Disadvantaged'], errors='coerce')
economic = pd.merge(ethnicity17, economic17, on=['Division Name', "School Name"])
economic_per = economic
economic_per['Total Disadvantaged'] = (economic['Total Disadvantaged']/economic['Totals'])
economic_per = economic_per.drop("Totals", axis=1)
#print(list(economic_per))
#print(economic_per.head(5))

#----------Data------------#
data = pd.merge(score17, ethnic_per, on=['Division Name', 'School Name'])
data = pd.merge(data, economic_per,  on=['Division Name', "School Name"])
data = data.drop(["School Name", "White"], axis=1)
print(data.head(5))



