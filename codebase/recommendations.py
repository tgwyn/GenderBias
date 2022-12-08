#Tony Gwyn
#12/7/22

# A dictionary of movie critics and their ratings of a small
# set of movies
critics={'Lisa Rose': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5,
 'Just My Luck': 3.0, 'Superman Returns': 3.5, 'You, Me and Dupree': 2.5, 
 'The Night Listener': 3.0},
'Gene Seymour': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5, 
 'Just My Luck': 1.5, 'Superman Returns': 5.0, 'The Night Listener': 3.0, 
 'You, Me and Dupree': 3.5}, 
'Michael Phillips': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0,
 'Superman Returns': 3.5, 'The Night Listener': 4.0},
'Claudia Puig': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0,
 'The Night Listener': 4.5, 'Superman Returns': 4.0, 
 'You, Me and Dupree': 2.5},
'Mick LaSalle': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0, 
 'Just My Luck': 2.0, 'Superman Returns': 3.0, 'The Night Listener': 3.0,
 'You, Me and Dupree': 2.0}, 
'Jack Matthews': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
 'The Night Listener': 3.0, 'Superman Returns': 5.0, 'You, Me and Dupree': 3.5},
'Toby': {'Snakes on a Plane':4.5,'You, Me and Dupree':1.0,'Superman Returns':4.0}}


from math import sqrt

# Returns a Euclidean distance-based similarity 
# score for person1 and person2
#
def sim_distance(prefs, person1, person2):
  # Get the list of shared_items: the names of the movies that both
  #  persons rated.
  si=[]
  for item in prefs[person1]:
    if item in prefs[person2]: si.append(item)

  # if they have no ratings in common, return 0
  if len(si)==0: return 0

  # Add up the squares of all the differences
  sum_of_squares=0.0
  for item in si:
    sum_of_squares += (prefs[person1][item] - prefs[person2][item]) ** 2

  # Distance = square root of sum of squares
  dist = sqrt(sum_of_squares)

  # Invert so that closer (shorter distance) becomes better (bigger numbers)
  return 1/(1+dist)


# Pearson similarity between two movie raters
# (or between two movies)
#
def sim_pearson(prefs,p1,p2):
  x_list, y_list = extract_item_vals(prefs, p1, p2)
  if len(x_list) == 0:
    return 0.0
  return pearson_r(x_list, y_list)


# Extract parallel lists of shared items between
# two people. (For example, all the ratings of movies
# that both people have rated.)
#
def extract_item_vals(prefs, p1, p2):
  
  # Get the list of movies that both p1 and p2 rated
  si=[]  # si = shared items
  p1_movies = list(prefs[p1])  # Movies that p1 rated
  p2_movies = list(prefs[p2])  # Movies that p2 rated
  for item in p1_movies:
    if item in p2_movies:
        si.append(item)


  # Create two parallel lists of the common values.
  x_list = [prefs[p1][it] for it in si]
  y_list = [prefs[p2][it] for it in si]
  return (x_list, y_list)


# Returns the Pearson correlation coefficient between
# two lists of numbers.
#
# This computation is less numerically sensitive than the
# one in the Programming Collective Intelligence book.
#
def pearson_r(x_list, y_list):
  n = len(x_list)
  if n < 2 or not (n == len(y_list)):
    return 0.0
  
  # Means and variances of each list
  x_bar = float(sum(x_list))/n
  y_bar = float(sum(y_list))/n
  x_s = sqrt(online_variance(x_list))
  y_s = sqrt(online_variance(y_list))

  # If standard deviations are relatively small (or zero),
  #  division is dangerous or undefined.  And r is not
  #  meaningful anyway in that case.
  if x_s <= abs(0.000001*x_bar) or y_s <= abs(0.000001*y_bar):
    return 0.0

  # Pearson correlation: (1/(n-1)) * sum[ (xi-mu_x)/s_x * (y_i-mu_y)/s_y ]
  r = 0.0
  for xi, yi in zip(x_list, y_list):
     r += ((xi - x_bar)/x_s) * ((yi - y_bar)/y_s)
  return (1.0/(n-1))*r


# Numerically stable method of comuting variance using
# Welford's method. This code was obtained from the Wikipedia
# article: Algorithms for caluclating variance.
#
def online_variance(data):
    mean = 0.0
    M2 = 0.0
    for i, x in enumerate(data):
        delta = x - mean
        mean = mean + delta/(i+1)
        M2 = M2 + delta*(x - mean)  # This expression uses the new value of mean
    return M2/(len(data) - 1)

# Returns the best matches for person from the prefs dictionary. 
# Number of results and similarity function are optional params.
def topMatches(prefs,person,n=5,similarity=sim_pearson):
  scores = []

  for other in prefs:
    if other != person:
      scores.append( (similarity(prefs, person, other), other) )
                    
  scores.sort()
  scores.reverse()
  return scores[0:n]

# Gets recommendations for a person (me) by using a weighted average
# of every other user's rankings
def getRecommendations(prefs,person,similarity=sim_pearson):
  totals={}
  simSums={}

  # Loop over all reviewers
  for other in prefs:
    # don't compare me to myself
    if other==person: continue

    # Similiarity between me and other
    sim=similarity(prefs,person,other)

    # ignore scores of zero or lower
    if sim<=0: continue

    # For all the other user's ratings...
    for item in prefs[other]:
	    
      # only score movies I haven't seen yet
      if item not in prefs[person] or prefs[person][item]==0:

        # Similarity * Score
        totals.setdefault(item,0)
        totals[item]+=prefs[other][item]*sim

        # Sum of similarities
        simSums.setdefault(item,0)
        simSums[item]+=sim

  # Create the normalized list
  rankings=[]
  for item,total in totals.items():
     rankings.append((total/simSums[item], item))

  # Return the sorted list
  rankings.sort()
  rankings.reverse()
  return rankings

def transformPrefs(prefs):
  result={}
  for person in prefs:
    for item in prefs[person]:
      result.setdefault(item,{})
      
      # Flip item and person
      result[item][person]=prefs[person][item]
  return result


def calculateSimilarItems(prefs,n=10):
  # Create a dictionary of items showing which other items they
  # are most similar to.
  result={}
  # Invert the preference matrix to be item-centric
  itemPrefs=transformPrefs(prefs)
  c=0
  for item in itemPrefs:
    # Status updates for large datasets
    c+=1
    if c%100==0: print("%d / %d" % (c,len(itemPrefs)))
    # Find the most similar items to this one
    scores=topMatches(itemPrefs,item,n=n,similarity=sim_distance)
    result[item]=scores
  return result

def getRecommendedItems(prefs,itemMatch,user):
  userRatings=prefs[user]
  scores={}
  totalSim={}
  # Loop over items rated by this user
  for (item,rating) in userRatings.items( ):

    # Loop over items similar to this one
    for (similarity,item2) in itemMatch[item]:

      # Ignore if this user has already rated this item
      if item2 in userRatings: continue
      # Weighted sum of rating times similarity
      scores.setdefault(item2,0)
      scores[item2]+=similarity*rating
      # Sum of all the similarities
      totalSim.setdefault(item2,0)
      totalSim[item2]+=similarity

  # Divide each total score by total weighting to get an average
  rankings=[(score/totalSim[item],item) for item,score in scores.items( )]

  # Return the rankings from highest to lowest
  rankings.sort( )
  rankings.reverse( )
  return rankings

# Load a set of movie ratings from the MovieLens database.
#
# Named parameters:
#   path=  relative path to the directory where movielens data is stored
#   file=  file of movielens data
#   titles=  file of movielens titles (usually not needed to specify it)
#   prefs=  Put your current database here if you want to MERGE IN new
#            data, leave it blank if you want to create a separate new data set.
#
import os
def loadMovieLens(path='', file='u.data', titles='u.item', prefs=None):
  # Get movie titles
  movies={}
  for line in open(os.path.join(path, titles), encoding='iso-8859-1'):
    (id,title)=line.rstrip().split('|')[0:2]
    movies[id]=title
  
  # Load data
  if not prefs: prefs={}
  for line in open(os.path.join(path, file), encoding='iso-8859-1'):
    if len(line.strip()) == 0: continue   # Ignore blank lines
    # Break apart into four fields, using whitespace delimiter
    try:
        (user,movieid,rating,ts) = line.split()
    except Exception as inst:
        print('Error: ', inst)
        print('  Ignoring input line: ', repr(line))
        continue
    # Add this user's preference into the database
    if not user in prefs:
        prefs[user] = {}
    prefs[user][movies[movieid]]=float(rating)
  return prefs

#----------------------------------------
# Student's New Code Goes Here
#----------------------------------------

# Get one rating: this user rating for this movie. Base is a database.
# It will return -1 if the user is wrong, or the user did not rate this movie
def one_rating(base, user, movie):
    if not user in base:  # Check to see if this critic is in the database
        return -1
    # You finish this function here
    
#-------- 
# p = condProb(base, user1, user2)
#
# returns: prob(user1 saw a movie given that user2 saw the movie)
#
# Computed by: numerator/denominator, where:
#    numerator =   (count of movies both user1 and user2 rated) 
#    denominator = (count of movies only user2 rated)
#
def condProb(base, user1, user2):
   #...you write the rest
   #-- extract_item_vals contains example python statements to get
   #  a list of movies rated by a single user, and also get a list
   #  of movies that both users rated.
   
  return 1.0   ## REPLACE THIS WITH YOUR ACTUAL RETURN VALUE


# Adjusted Pearson similarity between two movie raters
# (or between two movies)
#
# You can call sim_pearson and condProb
def sim_pearson_adj(prefs,p1,p2):
    return 1.0  # REPLACE WITH ACTUAL COMPUTATION

# compare2(base, user1, user2)
#
# prints ratings for movies that both users have in common.
#
def compare2(base, user1, user2):
   #...you write the rest
   print("stars1 stars2 movie")     # REPLACE WITH YOUR OWN CODE
# Revision of 'recommendations.py' from Programming Collective Intelligence,
#  chapter 2.
#

# A dictionary of movie critics and their ratings of a small
# set of movies
critics={'Lisa Rose': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5,
 'Just My Luck': 3.0, 'Superman Returns': 3.5, 'You, Me and Dupree': 2.5, 
 'The Night Listener': 3.0},
'Gene Seymour': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5, 
 'Just My Luck': 1.5, 'Superman Returns': 5.0, 'The Night Listener': 3.0, 
 'You, Me and Dupree': 3.5}, 
'Michael Phillips': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0,
 'Superman Returns': 3.5, 'The Night Listener': 4.0},
'Claudia Puig': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0,
 'The Night Listener': 4.5, 'Superman Returns': 4.0, 
 'You, Me and Dupree': 2.5},
'Mick LaSalle': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0, 
 'Just My Luck': 2.0, 'Superman Returns': 3.0, 'The Night Listener': 3.0,
 'You, Me and Dupree': 2.0}, 
'Jack Matthews': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
 'The Night Listener': 3.0, 'Superman Returns': 5.0, 'You, Me and Dupree': 3.5},
'Toby': {'Snakes on a Plane':4.5,'You, Me and Dupree':1.0,'Superman Returns':4.0}}


from math import sqrt

# Returns a Euclidean distance-based similarity 
# score for person1 and person2
#
def sim_distance(prefs, person1, person2):
  # Get the list of shared_items: the names of the movies that both
  #  persons rated.
  si=[]
  for item in prefs[person1]:
    if item in prefs[person2]: si.append(item)

  # if they have no ratings in common, return 0
  if len(si)==0: return 0

  # Add up the squares of all the differences
  sum_of_squares=0.0
  for item in si:
    sum_of_squares += (prefs[person1][item] - prefs[person2][item]) ** 2

  # Distance = square root of sum of squares
  dist = sqrt(sum_of_squares)

  # Invert so that closer (shorter distance) becomes better (bigger numbers)
  return 1/(1+dist)


# Pearson similarity between two movie raters
# (or between two movies)
#
def sim_pearson(prefs,p1,p2):
  x_list, y_list = extract_item_vals(prefs, p1, p2)
  if len(x_list) == 0:
    return 0.0
  return pearson_r(x_list, y_list)


# Extract parallel lists of shared items between
# two people. (For example, all the ratings of movies
# that both people have rated.)
#
def extract_item_vals(prefs, p1, p2):
  
  # Get the list of movies that both p1 and p2 rated
  si=[]
  p1_movies = list(prefs[p1])  # Movies that p1 rated
  p2_movies = list(prefs[p2])  # Movies that p2 rated
  for item in p1_movies:
    if item in p2_movies:
        si.append(item)

  # Create two parallel lists of the common values.
  x_list = [prefs[p1][it] for it in si]
  y_list = [prefs[p2][it] for it in si]
  return (x_list, y_list)


# Returns the Pearson correlation coefficient between
# two lists of numbers.
#
# This computation is less numerically sensitive than the
# one in the Programming Collective Intelligence book.
#
def pearson_r(x_list, y_list):
  n = len(x_list)
  if n < 2 or not (n == len(y_list)):
    return 0.0
  
  # Means and variances of each list
  x_bar = float(sum(x_list))/n
  y_bar = float(sum(y_list))/n
  x_s = sqrt(online_variance(x_list))
  y_s = sqrt(online_variance(y_list))

  # If standard deviations are relatively small (or zero),
  #  division is dangerous or undefined.  And r is not
  #  meaningful anyway in that case.
  if x_s <= abs(0.000001*x_bar) or y_s <= abs(0.000001*y_bar):
    return 0.0

  # Pearson correlation: (1/(n-1)) * sum[ (xi-mu_x)/s_x * (y_i-mu_y)/s_y ]
  r = 0.0
  for xi, yi in zip(x_list, y_list):
     r += ((xi - x_bar)/x_s) * ((yi - y_bar)/y_s)
  return (1.0/(n-1))*r


# Numerically stable method of comuting variance using
# Welford's method. This code was obtained from the Wikipedia
# article: Algorithms for calculating variance.
#
def online_variance(data):
    mean = 0.0
    M2 = 0.0
    for i, x in enumerate(data):
        delta = x - mean
        mean = mean + delta/(i+1)
        M2 = M2 + delta*(x - mean)  # This expression uses the new value of mean
    return M2/(len(data) - 1)

# Returns the best matches for person from the prefs dictionary. 
# Number of results and similarity function are optional params.
def topMatches(prefs,person,n=5,similarity=sim_pearson):
  scores = []

  for other in prefs:
    if other != person:
      scores.append( (similarity(prefs, person, other), other) )
                    
  scores.sort()
  scores.reverse()
  return scores[0:n]

# Gets recommendations for a person (me) by using a weighted average
# of every other user's rankings
def getRecommendations(prefs,person,similarity=sim_pearson):
  totals={}
  simSums={}

  # Loop over all reviewers
  for other in prefs:
    # don't compare me to myself
    if other==person: continue

    # Similiarity between me and other
    sim=similarity(prefs,person,other)

    # ignore scores of zero or lower
    if sim<=0: continue

    # For all the other user's ratings...
    for item in prefs[other]:
	    
      # only score movies I haven't seen yet
      if item not in prefs[person] or prefs[person][item]==0:

        # Similarity * Score
        totals.setdefault(item,0)
        totals[item]+=prefs[other][item]*sim

        # Sum of similarities
        simSums.setdefault(item,0)
        simSums[item]+=sim

  # Create the normalized list
  rankings=[]
  for item,total in totals.items():
     rankings.append((total/simSums[item], item))

  # Return the sorted list
  rankings.sort()
  rankings.reverse()
  return rankings

def transformPrefs(prefs):
  result={}
  for person in prefs:
    for item in prefs[person]:
      result.setdefault(item,{})
      
      # Flip item and person
      result[item][person]=prefs[person][item]
  return result


def calculateSimilarItems(prefs,n=10):
  # Create a dictionary of items showing which other items they
  # are most similar to.
  result={}
  # Invert the preference matrix to be item-centric
  itemPrefs=transformPrefs(prefs)
  c=0
  for item in itemPrefs:
    # Status updates for large datasets
    c+=1
    if c%100==0: print("%d / %d" % (c,len(itemPrefs)))
    # Find the most similar items to this one
    scores=topMatches(itemPrefs,item,n=n,similarity=sim_distance)
    result[item]=scores
  return result

def getRecommendedItems(prefs,itemMatch,user):
  userRatings=prefs[user]
  scores={}
  totalSim={}
  # Loop over items rated by this user
  for (item,rating) in userRatings.items( ):

    # Loop over items similar to this one
    for (similarity,item2) in itemMatch[item]:

      # Ignore if this user has already rated this item
      if item2 in userRatings: continue
      # Weighted sum of rating times similarity
      scores.setdefault(item2,0)
      scores[item2]+=similarity*rating
      # Sum of all the similarities
      totalSim.setdefault(item2,0)
      totalSim[item2]+=similarity

  # Divide each total score by total weighting to get an average
  rankings=[(score/totalSim[item],item) for item,score in scores.items( )]

  # Return the rankings from highest to lowest
  rankings.sort( )
  rankings.reverse( )
  return rankings

# Load a set of movie ratings from the MovieLens database.
#
# Named parameters:
#   path=  relative path to the directory where movielens data is stored
#   file=  file of movielens data
#   titles=  file of movielens titles (usually not needed to specify it)
#   prefs=  Put your current database here if you want to MERGE IN new
#            data, leave it blank if you want to create a separate new data set.
#
import os
def loadMovieLens(path='', file='nu.data', titles='nu.item', prefs=None):
  global movies
  
  # Get movie titles
  movies={}
  
  # Kluge: the older data sets used iso-8859-1 encoding in movie titles
  enc = 'utf-8'
  
  #for line in open(os.path.join(path, titles), encoding='iso-8859-1'):
  for line in open(os.path.join(path, titles), encoding=enc):
    (id,title)=line.rstrip().split('|')[0:2]
    movies[id]=title
  
  # Load data
  if not prefs: prefs={}
  for line in open(os.path.join(path, file), encoding=enc):
    if len(line.strip()) == 0: continue   # Ignore blank lines
    # Break apart into four fields, using whitespace delimiter
    try:
        user,movieid,rating = line.split()[:3]
    except Exception as inst:
        print('Error: ', inst)
        print('  Ignoring input line: ', repr(line))
        continue
    # Add this user's preference into the database
    if not user in prefs:
        prefs[user] = {}
    prefs[user][movies[movieid]]=float(rating)
  return prefs

#----------------------------------------
# Answer code from first lab here
#----------------------------------------

# Get one rating: this user rating for this movie. Base is a database.
# It will return -1 if the user is wrong, or the user did not rate this movie
def oneRating(base, user, movie):
    if not user in base:  # Check to see if this critic is in the database
        return -1
    # You finish this function here
    if not movie in base[user]: 
        return -1
    return base[user][movie]
#
#-------- 
# p = condProb(base, user1, user2)
#
# returns: prob(user1 saw a movie given that user2 saw the movie)
#
# Computed by: numerator/denominator, where:
#    numerator =   (count of movies both user1 and user2 rated) 
#    denominator = (count of movies only user2 rated)
#
def condProb(base, user1, user2):
 
    # Get the list of movies that both p1 and p2 rated
  si=[]  # si = shared items
  p1_movies = list(base[user1])  # Movies that p1 rated
  p2_movies = list(base[user2])  # Movies that p2 rated

  for item in p1_movies:
    if item in p2_movies:
        si.append(item)
  return len(si) / len(p2_movies)

# An alternative way to get a list of shared items using list comprehension
#  si = [movie for movie in p1_movies if movie in p2_movies]


# Adjusted Pearson similarity between two movie raters
# (or between two movies)
#
# You can use sim_pearson and condProb
def sim_pearson_adj(prefs,p1,p2):
    return sim_pearson(prefs, p1, p2) * condProb(prefs, p1, p2)


 # compare2(base, user1, user2)
#
# prints ratings for movies that both users have in common.
#
def compare2(base, user1, user2):
   for movie in base[user1]:
       if movie in base[user2]:
           print(base[user1][movie], base[user2][movie], movie)

#--------------------------------------------------------------------
# STUDENT LAB2 CODE STARTS HERE
#
           
# Validate a test set against a training set.
# Print the RMS error.
#
#def train_and_test(trainfile='u1.base', testfile='u1.test'):
#  base = loadMovieLens(trainfile)
#  test = loadMovieLens(testfile)
#  evaluate(train, test)
#   <- REMOVE THIS TO UNCOMMENT THE SKELETON CODE. (Also at the end)

def evaluate(base, test, similarity=sim_pearson, trace=0):

  # Initialize sum and count for the RMS calculation, 
  rms_sum = 0
  cnt = 0

  # iterate over every user in the test set
  for user in test.keys():

    # Get the recommendations for that test user,
    #   using the 'base' (training) database,
    #   and the similarity function which was specified
    recs = getRecommendations(base, user, similarity=similarity)

    # Examine each recommendation for the test user 
    for predicted_rating, test_movie in recs:

      # Get the actual rating from the test database for the test user and movie
      actual_rating = oneRating(test,user,test_movie)
                                
      # Compute error only if user rated this movie in the test database
      if actual_rating != -1:

        # The actual rating - the predicted rating is the error
        err = actual_rating - predicted_rating
        errsq =  (err*err)  # square it

        # debugging write: show the error-squared
        if trace>0:
          print(user, test_movie, actual_rating, predicted_rating)
          trace -= 1

        # Sum up the squares of the error
        rms_sum += errsq
        cnt += 1

  # return the square root of the (sum / count)
  return sqrt(rms_sum/cnt)

# REMOVE THE THREE QUOTES TO UNCOMMENT THE SKELETON. (Also at the beginning)

