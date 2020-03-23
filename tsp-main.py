# this follows the example project here: https://towardsdatascience.com/evolution-of-a-salesman-a-complete-genetic-algorithm-tutorial-for-python-6fe5d2b3ca35
#basically just followed this all the way thru to get a feel for how it works.

import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as pyplot


#city class to create and handle cities, also method to return distance to another city
class City:
    def __init__(self, x, y):
        self.x=x
        self.y=y

    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y-city.y)
        distance = np.sqrt((xDis**2) + (yDis**2))
        return distance

    def __repr__(self):
        return "(" + str(self.x)+ "," + str(self.y + ")"

#fitness class to determine route length and fitness of an individual solution
class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def routeDistance(self):
        if self.distance ==0:
            pathDistance = 0
            for i in range(0, len(self.route)): #loops thru the route, adding distances using City.distance method
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i+1]
                else:
                    toCity = self.route[0] #loops back once end of route array is reached
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance
    
    def routeFitness(self): #takes inverse so we can minimize
        if self.fitness ==0:
            self.fitness = 1/float(self.routeDistance())
        return self.fitness

def createRoute(cityList): #generates a random route
    route = random.sample(cityList, len(cityList))
    return route

#generates a population of routes based on above, and input of cityList
def initialPopulation(popSize, cityList):
    population=[]

    for i in range(0,popSize):
        population.append(createRoute(cityList))
    return population

def rankRoutes(population):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True) #itemgetter is a passed function to sorted

#here we pick parents from the ranked routes, assigning weighted probability based on their ranking
def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum() # this is the routlette wheel calculation. Gives relative fitness weight by looking at cumulative "contribution" to total fitness sum. Ie, how much space does it take up on roulette wheel calc
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()  #assign fitness weights to each route (come back to this)

    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])   #this automatically picks the top routes to be sure to include them for mating
    for i in range(0, len(popRanked) - eliteSize) #the rest are selected based on randomly selected cutoff criteria. ie, mating pool size changes each time.
        pick = 100*random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i,3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults

def matingPool(population, selectionResults): #just generates the actual mating pool array from the selection results output
    matingpool = []
    for i in range(0, len(selectionRsults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool


