#D:\Python\367-64b\python.exe
import tensorExample as tensor
import sys,time,random
class chrom:
    def __init__(self,LR,maxE,L1,L2,L3,fitness = None):
        self.LR = LR
        self.maxE = maxE
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.fitness = fitness
    def deepCopy(self):
        return chrom(self.LR,self.maxE,self.L1,self.L2,self.L3,self.fitness)
    def calculateFitness(self):
        error,epochs = tensor.nn(self.LR,self.maxE,self.L1,self.L2,self.L3)
        #print(error,epochs)
        #Fitness is found by taking error as a multiplier against a flat bonus and subtracting out time taken
        self.fitness = float((100*error)-epochs)
    def print(self,num):
        print("#"+str(num)+ "|Fitness="+str(self.fitness) + "|LR="+ str(self.LR) + "|Max Epoch ="+str(self.maxE) + "|L1=" + str(self.L1) + "|L2="+ str(self.L2)+"|L3="+str(self.L3))
def initPopulation(PopSize):
    Population = []
    for x in range(PopSize):
        Population.append(chrom(random.uniform(.001,.01),random.randint(1,5000),random.randint(800,5000),random.randint(800,5000),random.randint(800,5000)))
    return Population
def printPopulation(pop,gen = 0):
    numCounter = 0
    print("=========Generation "+ str(gen) +"==============")
    for c in pop:
        c.print(numCounter)
        numCounter+=1
    print("============================================")
#Input param
if (len(sys.argv)== 4):
    #use input params
    PopSize = sys.argv[1]
    crossRate = sys.argv[2]
    mutRate = sys.argv[3]
elif (len(sys.argv)==1):
    #No input param, use defaults
    PopSize = 10
    crossRate = .5
    mutRate= .01
else:
    #quit
    print("Run as: python ga.py N cR mR")
    quit()
#=================================
population = initPopulation(PopSize)
printPopulation(population)


#test = chrom(.003,500,2432,2322,1943)
#test.calculateFitness()
#test.print()
#print(tensor.nn())