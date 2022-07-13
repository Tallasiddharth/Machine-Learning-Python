

@author: NikhilJ
"""

print("Hello World")
import numpy as np
import pandas as pd
# Strings
str1 = "Hi. I am Nikhil. "
str2 = "How are you?"
str21= str1[::-1]
str21

str1[-4]+str1[-3]+str2[8:11]
print(str1 + str2)
str3 = str1[0:3]+str1[-8:-2] #cancatenation operation with slicing
print(str3)
str3[1]*2
type(str3) 
ls22 =list(str1)
ls22[1]='I'
print(str(ls22))
str4 = "Hi.Nikhil"
ls01=list(str1)
print(set(str1))
ls01
newstr =''.join(ls01)
newstr
print(tuple(str1))
# Lists

ls_1 = [1, 2, 3] # list defined
type(ls_1) #type identification
ls_1.append(4.5) # add an element 4.5 to list
ls_1
ls_1.append("4") # add an element "4" to list ls_1
ls_1
ls_1.remove(4.5) # removes 4.5 from list ls_1
ls_1
ls_2 =[4,5,6]    
ls_1.extend(ls_2) #list extention
ls_1.remove("4")
ls_1=ls_1[::-1] #list reversal
ls_3 = ls_1 + ls_2 #List cancatenation
ls_3[5]=0 #mutation
ls_3
ls_5=['Anil','Aashish','Aadithya','Aathiya','Aadhya','Aavantika']
ls_5.append('Aanandita')
ls_5.sort()
ls_5
s= sum(ls_1,10)
s
ls_1.insert(2,45)
ls_1
ls_6 =[1,2,3,4,5,6,7,8,9]
i =iter(ls_6)
dict2=dict(zip(i,i))
dict2
dict1=dict(zip(ls_6[1::2],ls_6[0::2])) #zip operation to create a dictionary
print(dict1)
ls_7=zip(*[iter(ls_6)]*3) #splitting a list into pair of 3
list(ls_7)
set1=set(ls_6)
set1

# Tuples

t1 = (5,6,"true")
t1
t2 = (7,8, "False")
"False" in t2 # check the content of tupple
"FaLse" in t2 # case sensitivity
t3 = t1+t2 # concatenation
t4 =(t1[1],t2[2]) # Extraction to form a new
t4
ls_4 =list(t4) # Convert tuple to list
ls_4
t5 = tuple(ls_4) # convert list to tuple
t6 = (145/7,168/9,174/4,219/3,203/11)
len(t6)
print(max(t6)) #finding max value in tuple

# Dictionary
# Syntax d_var ={key1:value1, key2:val2,key3=val3....etc}

d ={1:'Nikhil',2:'Vinayak',3:'Anirudha',4:'Sunil',5:'Surekha',6:'Shyam'}
print (d)
d.copy() #create a temporary copy of dictionary
d.up
d
d.keys() # to view all keys in dictionary
d.items() # values associated alongside keys res
d.values() # values present in ordered key  format
d[1]  # data retrieve with key
d[1]='vin' #data mutation
d[1]='Nikhil' #reassign mutation
#  fromkeys() creates a new dictionary with keys from seq and values set to value.
# syntax d_var.fromkeys(seq,values)
d.get('Education','Not Valid Key')
print(d)


g =list()
print(g)
str7 ='Nikhil'
print(list(str7))

divmod(10,6)
result =set('12')
print(result)
result2 =set('javaclick')
result2
newstr1 =''.join(result2)
newstr1
a ='12'
print(set(a))
import pandas as pd
d1 = {'Name' : ['JohnnieWalker','Chivas Regal','Amrut','RedLabel'], 'Age': [40,30,12,45]}
print(d1)

df1=pd.DataFrame(d1,columns=['Name','Age'])

d2 = {'Name' : ['Johnnie Walker','Chivas Regal','The Dalmore','Amrut','Glein Fiddich'], 'Storage': [25,35,50,30,58]}
print(d2)
df2=pd.DataFrame(d2,columns=['Name','Storage'])
df1
