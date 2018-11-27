#D:\Python\367-64b\python.exe
import tensorExample as tensor
import sys,time,random,csv
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
            print("Fitness for "+str(counter) + "= " + str(p.fitness))
            counter += 1
            total += p.fitness
            if (p.fitness<self.min):
                self.min = p.fitness
            if (p.fitness>self.max):
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
    def tournSelection(self):
        newPop = []
        a,b = None,None
        for x in range(len(self.pop)):
            y = random.randint(0,len(self.pop)-1)
            a = self.pop[y]
            y = random.randint(0,len(self.pop)-1)
            b = self.pop[y]
            if a.fitness > b.fitness:
                newPop.append(a)
            else:
                newPop.append(b)
        self.pop = newPop
    def crossover(self,crossRate):
        temp1,temp2 = None,None
        newPop = []
        random.shuffle(self.pop)
        for c in self.pop:
            if temp1 == None:
                temp1 = c.deepCopy()
                continue
            temp2 = c.deepCopy()
            #roll for crossover
            if (random.uniform(0.0,1.0)<crossRate):
                #crossover hit
                #50% chance to swap each value
                #LR
                if (random.uniform(0.0,1.0)<.5):
                    x = temp1.LR
                    temp1.LR = temp2.LR
                    temp2.LR = x
                #maxEp
                if (random.uniform(0.0,1.0)<.5):
                    x = temp1.maxE
                    temp1.maxE = temp2.maxE
                    temp2.maxE = x
                #L1
                if (random.uniform(0.0,1.0)<.5):
                    x = temp1.L1
                    temp1.L1 = temp2.L1
                    temp2.L1 = x
                #L2
                if (random.uniform(0.0,1.0)<.5):
                    x = temp1.L2
                    temp1.L2 = temp2.L2
                    temp2.L2 = x
                #L3
                if (random.uniform(0.0,1.0)<.5):
                    x = temp1.L3
                    temp1.L3 = temp2.L3
                    temp2.L3 = x
            newPop.append(temp1)
            newPop.append(temp2)
            temp1,temp2 = None,None
        #odd population, just add remainder
        if (temp1 !=None):
            newPop.append(temp1)
        self.pop = newPop
    def mutation(self,mutRate):
        for c in self.pop:
            #Roll for mutation
            if (random.uniform(0.0,1.0)<mutRate):
                #Mutate everything
                c.LR = abs(c.LR + random.uniform(-.005,.005))
                c.maxE = abs(c.maxE + random.randint(-100,100))
                c.L1 = abs(c.L1 + random.randint(-100,100))
                c.L2 = abs(c.L2 + random.randint(-100,100))
                c.L3 = abs(c.L3 + random.randint(-100,100))

                





#==============Main=========================
#Input param
if (len(sys.argv)== 5):
    #use input params
    PopSize = int(sys.argv[1])
    genAmount = int(sys.argv[2])
    crossRate = float(sys.argv[3])
    mutRate = float(sys.argv[4])
elif (len(sys.argv)==1):
    #No input param, use defaults
    PopSize = 4
    genAmount = 3
    crossRate = .5
    mutRate= .01
else:
    #quit
    print("Run as: python ga.py N maxGen cR mR")
    quit()
#============File Prep=====================
filename = "N" + str(PopSize) +"Gen" + str(genAmount) + "cR" + str(int(crossRate*100)) + "mR" + str(int(mutRate*100))
file = open(filename+'.csv','wb',encoding='utf-8')
fileWriter = csv.writer(file,dialect='excel')
fileWriter.writerow(['Min','Avg','Max'])
#============Init Population=================
P = Population()
P.initPopulation(PopSize)
fileWriter.writerow([P.min,P.avg,P.max])
P.pop.sort(key=lambda x:x.fitness,reverse = True)
Best = P.pop[0].deepCopy()
BestGen = -1
P.print(-1)
for generation in range(genAmount+1):
    P.tournSelection()
    P.crossover(crossRate)
    P.mutation(mutRate)
    P.updateStats()
    fileWriter.writerow([P.min,P.avg,P.max])
    P.pop.sort(key=lambda x:x.fitness,reverse = True)
    P.print(generation)
    if (Best.fitness< P.pop[0].fitness):
        Best = P.pop[0].deepCopy()
        BestGen = generation
    if Best.fitness > 60:
        break
#Done
file.close()
print("====================Best Chrom found in gen:" + str(BestGen) + "========================")
Best.print(0)
