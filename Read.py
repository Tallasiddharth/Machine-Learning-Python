

@author: Sidd
#-------------------------Reading & Writing data in Files----------------------

import pandas

# Reading CSV Files with Pandas:
    
df = pandas.read_csv('D:/skillfiles/files/User_Data.csv')
print(df)

# Writing CSV Files with Pandas:
df.to_csv('D:/skillfiles/files/User_Data.csv')

# Reading Excel Files with Pandas
df1 = pandas.read_excel('D:/skillfiles/files/User_Data.xlsx')

df1 = pandas.read_excel('D:/skillfiles/files/User_Data.xlsx')
