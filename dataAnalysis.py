import string

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

from collections import OrderedDict

specialChar = "!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?"

def get_box_plot_data(labels, bp):
    rows_list = []

    for i in range(len(labels)):
        dict1 = {}
        dict1['label'] = labels[i]
        dict1['lower_whisker'] = bp['whiskers'][i*2].get_ydata()[1]
        dict1['lower_quartile'] = bp['boxes'][i].get_ydata()[1]
        dict1['median'] = bp['medians'][i].get_ydata()[1]
        dict1['upper_quartile'] = bp['boxes'][i].get_ydata()[2]
        dict1['upper_whisker'] = bp['whiskers'][(i*2)+1].get_ydata()[1]
        rows_list.append(dict1)

    return pd.DataFrame(rows_list)


def find_outliers_IQR(df):
    q1=df.quantile(0.25)
    q3=df.quantile(0.75)
    IQR=q3-q1
    # outliers = df[((df<(q1-1.5*IQR)) | (df>(q3+1.5*IQR)))]
    return ((q1-1.5*IQR), (q3+1.5*IQR))

def filter_abberent(occurences, initialData) :

    abberant_values = find_outliers_IQR(occurences)

    filteredData = initialData.loc[lambda x : (x.password_count < abberant_values[1]) & (x.password_count > abberant_values[0])]
    
    return filteredData


all_letters = string.ascii_letters + string.digits + string.punctuation

char_array = list(all_letters)

occur_array = [0] * len(char_array)

password_data = []

with open('./data/AshleyMadison.txt', encoding='utf-8') as some_file:
    for line in some_file:
        
        formated_line = line.strip().replace(" ","")
        password_data.append(tuple((formated_line,len(formated_line))))
        
        for char in formated_line:
            try : 
                index = char_array.index(char)
                occur_array[index]+=1
                
            except ValueError as err :
                print(err)
                print(ord(char))

password_data_idx, password_data_values = zip(*password_data)

password_length_np = (np.array(password_data_values))

#print(password_length_array)

#data = np.array(char_array,occur_array)

# df = pd.DataFrame({'lettre': char_array,
#                    'occurence': occur_array})

#print(df)

occurencesDic = dict(zip(char_array, occur_array))

#print(d)


regexContainsSpecialChar = "["+specialChar+"]"

regexContainsOnlySpecialChar = "^["+specialChar+"]+$"

specialCharOccurance = dict((k, v) for k, v in occurencesDic.items() if k in specialChar)

print(specialCharOccurance)

passwordContainingSpecialChar = list(filter(lambda v: re.match(regexContainsSpecialChar, v), password_data_idx))

print("Number of password containing special char :",len(passwordContainingSpecialChar))

passwordContainingOnlySpecialChar = list(filter(lambda v: re.match(regexContainsOnlySpecialChar, v), password_data_idx))

print("Number of password only containing special char : ",len(passwordContainingOnlySpecialChar))

print("Number of total password : ",len(password_data_idx))

figure, axis = plt.subplots(1, 3)

labels = ['data']

bp1 = axis[0].boxplot(occur_array)
axis[0].set_title("occurences des lettres")

print(get_box_plot_data(labels, bp1))

bp2 = axis[1].boxplot(password_length_np)
axis[1].set_title("longeurs des mots de passe")

print(get_box_plot_data(labels, bp2))

password_df = pd.DataFrame({'password length': password_data_values})

filteredData = filter_abberent(password_df["password length"],pd.DataFrame({"password_count":password_data_values,"password":password_data_idx}))

# filteredData2 = filteredData[~filteredData['password'].str.match(regexContainsSpecialChar)]

filteredData2 = list(filter(lambda v: re.match("^[a-zA-Z0-9]*$", v), filteredData["password"]))

filteredData3 = list(filter(lambda v: re.match("^[a-z]*$", v), filteredData["password"]))

filteredData4 = list(filter(lambda v: re.match("^[A-Z]*$", v), filteredData["password"]))

filteredData5 = list(filter(lambda v: re.match("^[0-9]*$", v), filteredData["password"]))


bp3 = axis[2].boxplot(filteredData.password_count)
axis[2].set_title("longeurs des mots de passe filtrés")

print(np.mean(filteredData.password_count))

with open('./data/PasswordFiltered.txt', "w") as file:
    for line in filteredData2:
        file.write(line+"\n")

print("Number of total password :",len(password_data_idx))
print("Number of password without abberent values :",len(filteredData["password"]))
print("Number of password without abberent values and special char :",len(filteredData2))
print("Deleted passwords :",len(password_data_idx)-len(filteredData2))
print("Number of password with only lowercase char :",len(filteredData3))
print("Number of password with only uppercase char :",len(filteredData4))
print("Number of password with only numbers :",len(filteredData5))

plt.show()




