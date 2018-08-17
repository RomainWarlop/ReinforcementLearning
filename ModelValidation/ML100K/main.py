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

red55 = [182/255,25/255,36/255]
blue55 = [83/255,173/255,180/255]

#movies = pd.read_csv("/home/romain/Data/Input/ml-100k/u.item",sep="|",index_col=False, encoding = "ISO-8859-1",
#                   names=['movie_id','title','releasedate',
#                   'videoreleasedate','IMDbURL','unknown','Action',
#                   'Adventure','Animation','Children','Comedy','Crime',
#                   'Documentary','Drama','Fantasy','FilmNoir','Horror',
#                   'Musical','Mystery','Romance','SciFi','Thriller','War',
#                   'Western'])
#movies.to_gbq("movielens100k.movies","data-science-55")

#path = '/home/romain/Documents/BitBucket/PhD/linucrl/ModelValidation/ML100K/'
path = "/home/romain/Bureau/ongoingImages/"

windows = [10] #[5,10,15]
genres = ["Action","Comedy","Adventure","Thriller","Drama"]
D = list(range(1,10))

tuples = list(itertools.product(*[windows,genres,D]))
index = pd.MultiIndex.from_tuples(tuples, names=['w','genre','d'])
out = pd.DataFrame(columns=['R2'],index=index)
out = out.sortlevel(0)
out = out.fillna(0)

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
        
        output = """SELECT weight, count(*) as n, AVG(rating) rating, STDDEV(rating) std_rating 
        FROM ("""+output+""") 
        WHERE """+genre+"""=1  
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
                out.loc[(w,genre,d),'R2'] = R2
                
                if d==5:
                    plt.figure(figsize=(8,6))
                    ax = plt.subplot(111)
                    ax.spines["top"].set_visible(False)
                    ax.spines["bottom"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.spines["left"].set_visible(False)
                    plt.xticks(fontsize=15)
                    plt.yticks(fontsize=15)
                    #plt.plot(data['weight'],data['rating'],'b-',label='historical ratings')
                    plt.errorbar(data['weight'],data['rating'],yerr=data['conf'],
                                 linewidth=2,label='historical ratings',ecolor=blue55,color=blue55)
                    plt.plot(data['weight'],mod.predict(data),'-',label='prediction',linewidth=2,c=red55)
                    #plt.title('Average reward with 1/t decrease for '+genre+' w='+str(w),fontsize=15)
                    plt.legend(loc=3,fontsize=25)
                    #plt.savefig(path+'w'+str(w)+'/1_t_'+genre+' w='+str(w)+'.png')
                    plt.savefig(path+genre+'.pdf')

#out = out.reset_index()
#out = pd.pivot_table(out,values='R2',index=['w','genre'],columns='d').reset_index()
#out.to_csv(path+'R2.csv',index=False)

