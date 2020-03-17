path = r'C:\Users\... file path'

allFiles = glob.glob(path + "/*.csv")

frame = pd.DataFrame()

df_list = []

for file in allFiles:
    df = pd.read_csv(file, index_col=None, header=0)
    df_list.append(df)
frame = pd.concat(df_list)   # ignore_index=True)
