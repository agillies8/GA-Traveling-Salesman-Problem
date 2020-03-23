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

#breeding - need to visit every city at least once, remember. need 'ordered crossover' see fig in medium post. pick subsegment of parent 1, put in same location, then fill in rest in order from 2nd parent, skipping pre selected cities
def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []

    geneA = int(random.random()* len(parent1)) #this could be done better
    geneB = int(random.random()*len(parent1))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])
    
    childP2 = [item for item in parent2 if item not in childP1] #look into whats happening here

    child = childP1 + childP2
    return child

def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize #sets array size for new children, less our elite that we will pass on
    pool = random.sample(matingpool, len(matingpool)) #randomly arrange parents from mating pool into pool

    for i in range(0,eliteSize): #pass thru the elite parents
        children.append(matingpool[i])

    for i in range(0,length): # breed everyone and fill out children using length above
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children

#swap mutation -this function is for individual. Note: cant just swtich, need to maintain list of all cities
def mutate(individual, mutationRate):
    for swapped in range(len(individual)): #cycle thru cities in the individual route
        if(random.random()< mutationRate): #if porbability is met, do the below and swap cities
            swapWith = int(random.random()*len(individual)) #pick random location to do the swap

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1
return individual

#cycle thru population and perform individual mutation function on each one
def mutatePopulation(population, mutationRate):
    mutatedPop=[]

    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop

def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children  =breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration

def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations): #to get the best route output
    pop = initialPopulation(popSize, population)
    print("initial distance: " + str(1/rankRoutes(pop)[0][1]))

    for i in range(0, generations):
        pop = nextGeneration(pop,eliteSize, mutationRate)

    print("Final distance: " = str(1 / rankRoutes(pop)[0][1]) )
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute

def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations): #if you want to plot the results
    pop = initialPopulation(popSize, population)

    progress = []
    progress.append(1/rankRoutes(pop)[0][1])

    for i in range(0, generations):
        pop = nextGeneration(pop,eliteSize, mutationRate)
        progress.append(1/rankRoutes(pop)[0][1])

    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show



cityList = []

for i in range(0,25):
    cityList.append(City(x=int(random.random()*200), y=int(random.random()*200)))

geneticAlgorithm(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)




