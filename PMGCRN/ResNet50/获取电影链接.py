import pandas as pd

df = pd.read_csv("./ml-1m/ml-1m/movies.dat", sep="::", header=None, names=["movie_id", "movie_name", "features"])

with open('./ml-1m/ml-1m/movies_link.txt', 'w') as f:
    for name in df.values:
        mname = 'https://www.imdb.com/find/?q='+str(name[1])
        print(name[1])
        f.write(mname+'\n')
f.close()
