#D:\Python\367-64b\python.exe
import tensorExample as tensor
import sys,time,random
class chrom:
    def __init__(self,LR,maxE,L1,L2,L3,fitness = None):
        #Learning Rate
        self.LR = LR
        #Max Epochs
        self.maxE = maxE
        #Nodes per layer
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        #Fitness
        self.fitness = fitness
    def deepCopy(self):
        return chrom(self.LR,self.maxE,self.L1,self.L2,self.L3,self.fitness)
    def calculateFitness(self):
        error,epochs = tensor.nn(self.LR,self.maxE,self.L1,self.L2,self.L3)
        #Fitness is found by taking error as a multiplier against a flat bonus and subtracting out time taken
        self.fitness = float((100*error)-epochs)
    def print(self,num):
        #Prints out all chrome data
        print("#"+str(num)+ "|Fitness="+str(self.fitness) + "|LR="+ str(self.LR) + "|Max Epoch ="+str(self.maxE) + "|L1=" + str(self.L1) + "|L2="+ str(self.L2)+"|L3="+str(self.L3))
class Population:
    def __init__(self):
        #List of chromes
        self.pop = []
        #Statisics
        self.min = 10000000
        self.max = -10000000
        self.avg = 0
    def initPopulation(self,PopSize):
        Population = []
        total = 0
        counter = 0
        #Max PopSize chroms of randomized values, add to list
        for x in range(PopSize):
            Population.append(chrom(random.uniform(.001,.05),random.randint(1,2000),random.randint(800,5000),random.randint(800,5000),random.randint(800,5000)))
        #Calculated the fitness of each, record statistics as fitness is updated
        for p in Population:
            print("==Getting Fitness for chrom:" + str(counter)+ "======")
            p.calculateFitness()
            print("Fitness for "+str(counter) + "=" + str(p.fitness))
            counter += 1
            total += p.fitness
            if (p.fitness<self.min):
                self.min = p.fitness
            elif (p.fitness>self.max):
                self.max = p.fitness
        self.avg = total/PopSize
        self.pop = Population
    def updateStats(self):
        #Reset Stats
        self.min = 10000000
        self.max = -10000000
        self.avg = 0
        total = 0
        counter = 0
        #Get fitness for each chrom, update statistics if needed
        for p in self.pop:
            print("==Getting Fitness for chrom:" + str(counter)+ "======")
            p.calculateFitness()
            print("Fitness for "+str(counter) + "=" + str(p.fitness))
            counter += 1
            total += p.fitness
            if (p.fitness<self.min):
                self.min = p.fitness
            elif (p.fitness>self.max):
                self.max = p.fitness
        self.avg = total/len(self.pop)
    def print(self,gen = 0):
        numCounter = 0
        print("=========Generation "+ str(gen) +"==============")
        for c in self.pop:
            c.print(numCounter)
            numCounter+=1
        print("============================================")
        print("Avg:" + str(self.avg) + "|Min:" + str(self.min) + "|Max:"+ str(self.max))
        print("============================================")

#==============Main=========================
#Input param
if (len(sys.argv)== 5):
    #use input params
    PopSize = sys.argv[1]
    genAmount = sys.argv[2]
    crossRate = sys.argv[3]
    mutRate = sys.argv[4]
elif (len(sys.argv)==1):
    #No input param, use defaults
    PopSize = 4
    genAmount = 2
    crossRate = .5
    mutRate= .01
else:
    #quit
    print("Run as: python ga.py N cR mR")
    quit()
#============Init Population=================
P = Population()
P.initPopulation(PopSize)

