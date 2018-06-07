#==============================================================================
# model validation on movielens100k dataset
# Plot of mean rating as a function of the penalty weight and fit using our model
#==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import itertools

path = '/home/romain/Documents/BitBucket/PhD/linucrl/ModelValidation/ML100K/'

windows = [10] #[5,10,15]
genres = ["Action","Comedy","Adventure","Thriller","Drama","Children","Crime",
          "Documentary","Romance","Horror","Musical","SciFi","War","Animation",
          "FilmNoir","Mystery","Western"]
d = 5

params = dict.fromkeys(genres)
for genre in genres:
    for w in windows:
        
        user50 = """SELECT user_id FROM
        (SELECT user_id, count(*) n FROM [data-science-55:movielens100k.ratings] GROUP BY 1)
        WHERE n>50
        """
        
        core = """
        SELECT 
            a.user_id user_id, b.movie_id movie_id, b.rating rating, b.timestamp timestamp
        FROM ("""+user50+""") as a
        LEFT JOIN [data-science-55:movielens100k.ratings] as b
        ON a.user_id = b.user_id
        """
        
        dGenres = ", ".join(list(map(lambda x: "d."+x+" "+x,genres)))
        core_genres = """
        SELECT c.user_id user_id, c.rating rating, c.timestamp timestamp, """+dGenres+""" 
        FROM ("""+core+""") as c 
        LEFT JOIN [data-science-55:movielens100k.movies] as d
        ON c.movie_id = d.movie_id
        """
        
        lag = "SELECT user_id, timestamp, "+genre+", rating"
        
        for i in range(1,w+1):
            lag += ", LAG("+genre+","+str(i)+") OVER (PARTITION BY user_id ORDER BY timestamp ASC) as genre_"+str(i)
        
        lag += """ FROM ("""+core_genres+""")"""
        
        weight = "SELECT "+genre+", rating"
        
        for i in range(1,w+1):
            weight += ", IF(genre_"+str(i)+"=1,1/"+str(i)+",0) as weight_"+str(i)
        
        weight += " FROM ("+lag+")"
            
        output = "SELECT "+genre+", rating, ROUND(weight_1"
        
        for i in range(2,w+1):
            output += "+weight_"+str(i)
        
        output += ",1) as weight FROM ("+weight+")"
        
#        output = """SELECT weight, count(*) as n, AVG(rating) rating, STDDEV(rating) std_rating 
#        FROM ("""+output+""") 
#        WHERE """+genre+"""=1  
#        GROUP BY 1"""
        
        data = pd.read_gbq(output,project_id="data-science-55")
        #data['conf'] = 1.96*data['std_rating']/np.sqrt(data['n'])
        if len(data)>10000:
            #data = data.loc[data['n']>100,]
            #data = data.sort_values('weight')
            
            for j in range(2,d+1):
                data['weight'+str(j)] = data['weight']**j
                data['weight'+str(j)] = data['weight']**j
            
            formula = "rating ~ weight"
            for j in range(2,d+1):
                formula += " + weight"+str(j)
                
            mod = smf.ols(formula,data).fit()
            R2 = int(np.round(mod.rsquared,2)*100)/100
            #out.loc[(w,genre,d),'R2'] = R2
            params[genre] = list(np.round(mod.params,2))
            print('*'*20)
            print(genre)
            print(mod.params)
            print('*'*20)
                
#out = out.reset_index()
#out = pd.pivot_table(out,values='R2',index=['w','genre'],columns='d').reset_index()
#out.to_csv(path+'R2.csv',index=False)

