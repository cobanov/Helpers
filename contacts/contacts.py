import pandas as pd 
import glob
import os
import sqlite3



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

def create_database():
	conn = sqlite3.connect('contacts.db')
	print ("Opened database successfully")

	conn.execute('CREATE TABLE students (user_name TEXT, name TEXT, file_name TEXT)')
	print ("Table created successfully")
	conn.close()



def main():
	file_name = "publisher.txt"
	content = read_files(file_name)
	user_names, names = get_contacts(content)
	df_contacts = write_dataframe(user_names, names, file_name)
	print(df_contacts.head())


	


main()