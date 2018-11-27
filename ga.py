#D:\Python\367-64b\python.exe
import tensorExample as tensor

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
    def print(self):
        print("Fitness="+str(self.fitness) + "|LR="+ str(self.LR) + "|Max Epoch ="+str(self.maxE) + "|L1=" + str(self.L1) + "|L2="+ str(self.L2)+"|L3="+str(self.L3))
    
    
#test = chrom(.003,500,2432,2322,1943)
#test.calculateFitness()
#test.print()
#print(tensor.nn())