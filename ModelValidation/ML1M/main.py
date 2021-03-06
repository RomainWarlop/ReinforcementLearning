#==============================================================================
# model validation on movielens1M dataset
# Plot of mean rating as a function of the penalty weight and fit using our model
#==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import itertools

#ratings = pd.read_csv("/home/romain/Data/Input/ml-1M/ratings.dat",sep="::",
#                   header=None,names=['user_id','movie_id','rating','timestamp'])
#ratings.to_gbq("movielens1M.ratings","data-science-55")

#movies = pd.read_csv("/home/romain/Data/Input/ml-1M/movies.dat",sep="::",
#                   header=None,names=['movie_id','title','genres'])
#movies.to_gbq("movielens1M.movies","data-science-55")

path = '/home/romain/Documents/BitBucket/PhD/linucrl/ModelValidation/ML1M/'

windows = [5,10,15]
genres = ["'Action'","'Comedy'","'Adventure'","'Thriller'","'Drama'"]
_genres = list(map(lambda x: str.replace(x,"'",""),genres))
D = list(range(1,10))

tuples = list(itertools.product(*[windows,_genres,D]))
index = pd.MultiIndex.from_tuples(tuples, names=['w','genre','d'])
out = pd.DataFrame(columns=['R2'],index=index)
out = out.sortlevel(0)
out = out.fillna(0)


for genre in genres:
    _genre = str.replace(genre,"'","")
    for w in windows:
        
        user50 = """SELECT user_id FROM
        (SELECT user_id, count(*) n FROM [data-science-55:movielens1M.ratings] GROUP BY 1)
        WHERE n>50
        """
        
        core = """
        SELECT 
            a.user_id user_id, b.movie_id movie_id, b.rating rating, b.timestamp timestamp
        FROM ("""+user50+""") as a
        LEFT JOIN [data-science-55:movielens1M.ratings] as b
        ON a.user_id = b.user_id
        """
        
        core_genres = """
        SELECT c.user_id user_id, c.rating rating, c.timestamp timestamp, d.genres genres
        FROM ("""+core+""") as c 
        LEFT JOIN [data-science-55:movielens1M.movies] as d
        ON c.movie_id = d.movie_id
        """
        
        lag = """SELECT user_id, timestamp, genres, rating"""
        
        for i in range(1,w+1):
            lag += ", LAG(genres,"+str(i)+") OVER (PARTITION BY user_id ORDER BY timestamp ASC) as genres_"+str(i)
        
        lag += """ FROM ("""+core_genres+""")"""
        
        weight = "SELECT genres, rating"
        
        for i in range(1,w+1):
            weight += ", IF(genres_"+str(i)+" CONTAINS "+genre+",1/"+str(i)+",0) as weight_"+str(i)
        
        weight += " FROM ("+lag+")"
            
        output = "SELECT genres, rating, ROUND(weight_1"
        
        for i in range(2,w+1):
            output += "+weight_"+str(i)
        
        output += ",1) as weight FROM ("+weight+")"
        
        output = """SELECT weight, count(*) as n, AVG(rating) rating, STDDEV(rating) std_rating
        FROM ("""+output+""") 
        WHERE genres CONTAINS """+genre+""" 
        GROUP BY 1"""
        
        data = pd.read_gbq(output,project_id="data-science-55")
        data['conf'] = 1.96*data['std_rating']/np.sqrt(data['n'])
        if data['n'].sum()>10000:
            data = data.loc[data['n']>100,]
            data = data.sort_values('weight')
            
            for j in range(2,max(D)+1):
                data['weight'+str(j)] = data['weight']**j
                data['weight'+str(j)] = data['weight']**j
            
            for d in D:
                print('d=',d)
                formula = "rating ~ weight"
                for j in range(2,d+1):
                    formula += " + weight"+str(j)
                    
                mod = smf.ols(formula,data).fit()
                R2 = int(np.round(mod.rsquared,2)*100)/100
                out.loc[(w,_genre,d),'R2'] = R2
                
                if d==5:
                    plt.figure(figsize=(8,6))
                    #plt.plot(data['weight'],data['rating'],'b-',label='historical ratings',linewidth=2)
                    plt.errorbar(data['weight'],data['rating'],yerr=data['conf'],linewidth=2,label='historical ratings')
                    plt.plot(data['weight'],mod.predict(data),'r-',label='prediction',linewidth=2)
                    plt.title('Average reward with 1/t decrease for '+_genre+' w='+str(w),fontsize=15)
                    plt.legend(loc='best')
                    plt.savefig(path+'w'+str(w)+'/1_t_'+_genre+' w='+str(w)+'.png')

out = out.reset_index()
out = pd.pivot_table(out,values='R2',index=['w','genre'],columns='d').reset_index()
out.to_csv(path+'R2.csv',index=False)
out.to_latex()

