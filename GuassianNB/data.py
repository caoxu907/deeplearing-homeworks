import pandas as pd

path = 'NBA_Season_Stats.csv'
data_df = pd.read_csv(path)
data_df = data_df.replace(' ', '').replace("C", "1.0").replace("PF", "2.0").replace("PG","3.0").replace("SG","4.0").replace("SF","5.0")
data_df = data_df.dropna()
data_df = data_df.reset_index(drop=True)
data_df = data_df[['G','MP','FG','FGA','FG%','3P','3PA','3P%','2P','2PA','2P%','eFG%','FT','FTA','FT%','ORB','DRB','TRB','AST','STL','BLK','TOV','PF','PTS','Pos']]
#data_df = data_df[['2P%', 'FT%', 'TRB', 'AST', 'STL', 'PTS','Pos','3P%']]
#data_df = data_df[:4000]
data_df.to_csv('data1.csv')
