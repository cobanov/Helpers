import glob
import os
import sqlite3
import pandas as pd 

# conn = sqlite3.connect('contacts.db')
# print ("Opened database successfully")

# conn.execute('CREATE TABLE students (name TEXT, addr TEXT, city TEXT, pin TEXT)')
# print ("Table created successfully")
# conn.close()

def read_files(file_name):
	with open(file_name) as f:
		content = f.readlines()

	content = [x.strip() for x in content]
	return content


def get_contacts(content):
	user_names = []
	names = []

	for i in content:
		
		user_name, name = i.split("-->")
		user_names.append(user_name.strip())
		names.append(name.strip())

	return user_names, names

def write_dataframe(user_names, names, file_name):	
	df_contacts = pd.DataFrame(columns=["User_Names", "Name", "Permissions"])
	df_contacts["User_Names"] = user_names
	df_contacts["Name"] = names
	df_contacts["Permissions"] = file_name
	return df_contacts



def main():
	files = glob.glob("*.txt")
	df_contacts = pd.DataFrame(columns=["User_Names", "Name", "Permissions"])

	for file_name in files:
		content = read_files(file_name)
		user_names, names = get_contacts(content)
		write_dataframe(user_names, names, file_name)
		print(df_contacts.head())


if __name__ == "__main__":
	main()