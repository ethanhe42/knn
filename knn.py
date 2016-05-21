import numpy as np
import sys

def readData(path,training=True):
    data_label=np.loadtxt(path,skiprows=1)
    data=dict()
    data['n']=len(data_label)
    if training==True:
        data['dim']=len(data_label[0])-1
        data['x'],y=np.hsplit(data_label,np.array([data['dim']]))
        data['y']=y.T[0].astype(int)
    else:
        data['dim']=len(data_label[0])
        data['x']=data_label
    return data

def arr2str(arr):
    s=''
    if len(arr.shape)==1:

        return ' '.join(map(str,arr))
    else:
        for subarr in arr:
            s+=arr2str(subarr)+'\n'
    return s

class knn:
    def __init__(self,train):
        self.train=train
    
    def predict(self, test, k):
        self.test=test
        # create a lookup table
        guess=[]
        for i in test['x']:
            # euclidean
            dist=np.sqrt(((i-self.train['x'])**2).sum(1))
            if k>self.train['n']:
                print 'warning k bigger than number of samples, k changed to n'
                k=self.train['n']
            #neighbors
            idx=dist.argsort()[:k]
            dist.sort()
            dist=dist[:k]
            klabels=self.train['y'][idx]
            
            val_counts=np.bincount(klabels)
            ranks=(-val_counts).argsort()
            label=ranks[0]
            minDist=dist[klabels==label].min()
            for i in range(1, len(ranks)-1):
                new_label=ranks[i]
                if val_counts[new_label]<val_counts[label]:
                    break
                new_dist=dist[klabels==new_label].min()
                # tie votes
                if new_dist<minDist:
                    minDist=new_dist
                    label=new_label
                # tie neighbor
                if new_dist==minDist and new_label<label:
                    label=new_label
            guess.append(label)
        self.guess=np.array(guess)
    
    def show(self):
        for i,j,k in zip(range(1, self.test['n']+1), self.test['x'], self.guess):
            print i, '.', arr2str(j), '--', str(k)
            
# main
k=int(sys.argv[1])
raw_train=sys.argv[2]
raw_test=sys.argv[3]
train=readData(raw_train)
test=readData(raw_test, False)
model=knn(train)
model.predict(test,k)
model.show()           
            
