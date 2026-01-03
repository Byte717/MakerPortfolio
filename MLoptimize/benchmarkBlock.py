import time
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import datasets,transforms
from torch.utils.data import DataLoader

from src.layers.BlockLinear import BlockLinear

class ClassicLinear(nn.Module):
    def __init__(self):
        super(ClassicLinear,self).__init__()
        self.network=nn.Sequential(
            nn.Linear(28*28,128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,10),
        )

    def forward(self,x):
        x=x.view(-1,28*28)
        return self.network(x)

class BlockedLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(28*28,126)
        self.relu=nn.ReLU()
        self.drop=nn.Dropout(0.2)
        self.fc2=BlockLinear(126,126)
        self.fc3=nn.Linear(126,10)

    def forward(self,x):
        x=x.view(-1,28*28)
        x=self.fc1(x)
        x=self.relu(x)
        x=self.drop(x)
        x=self.fc2(x)
        x=self.relu(x)
        x=self.fc3(x)
        return x

def getMnistLoaders(batchSize=256,numWorkers=2):
    tfm=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,)),
    ])
    trainDs=datasets.MNIST(root="./data",train=True,download=True,transform=tfm)
    testDs=datasets.MNIST(root="./data",train=False,download=True,transform=tfm)

    trainLoader=DataLoader(
        trainDs,
        batch_size=batchSize,
        shuffle=True,
        num_workers=numWorkers,
        pin_memory=False,
        persistent_workers=(numWorkers>0),
    )
    testLoader=DataLoader(
        testDs,
        batch_size=batchSize,
        shuffle=False,
        num_workers=numWorkers,
        pin_memory=False,
        persistent_workers=(numWorkers>0),
    )
    return trainLoader,testLoader

@torch.no_grad()
def evalAcc(model,loader):
    model.eval()
    correct=0
    total=0
    for x,y in loader:
        logits=model(x)
        pred=logits.argmax(dim=1)
        correct+=int((pred==y).sum().item())
        total+=y.numel()
    return correct/max(1,total)

def trainModel(model,trainLoader,testLoader,epochs=5,lr=1e-3):
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    lossFn=nn.CrossEntropyLoss()

    testAcc=[]
    epoch=0
    while epoch<epochs:
        model.train()
        for x,y in trainLoader:
            opt.zero_grad(set_to_none=True)
            logits=model(x)
            loss=lossFn(logits,y)
            loss.backward()
            opt.step()
        testAcc.append(evalAcc(model,testLoader))
        epoch+=1
    return testAcc

@torch.no_grad()
def benchmarkInference(model,loader,warmupBatches=10,benchBatches=200):
    model.eval()

    it=iter(loader)
    i=0
    while i<warmupBatches:
        try:
            x,_=next(it)
        except StopIteration:
            it=iter(loader)
            x,_=next(it)
        _=model(x)
        i+=1

    it=iter(loader)
    totalImages=0
    start=time.perf_counter()
    j=0
    while j<benchBatches:
        try:
            x,_=next(it)
        except StopIteration:
            it=iter(loader)
            x,_=next(it)
        _=model(x)
        totalImages+=x.size(0)
        j+=1

    end=time.perf_counter()
    dt=max(1e-9,end-start)
    ips=totalImages/dt
    msPerImg=(dt*1000.0)/max(1,totalImages)
    return ips,msPerImg

def plotAccuracyCurves(accClassic,accBlocked,outPath="mnist_accuracy.png"):
    plt.figure()
    x=list(range(1,len(accClassic)+1))
    plt.plot(x,accClassic,label="ClassicLinear")
    plt.plot(x,accBlocked,label="BlockedLinear")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("MNIST Test Accuracy vs Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outPath,dpi=200)

def plotInferenceBars(ipsClassic,ipsBlocked,outPath="mnist_inference_speed.png"):
    plt.figure()
    labels=["ClassicLinear","BlockedLinear"]
    values=[ipsClassic,ipsBlocked]
    plt.bar(labels,values)
    plt.ylabel("Images / second")
    plt.title("Inference Throughput (higher is better)")
    plt.tight_layout()
    plt.savefig(outPath,dpi=200)

def trainClassic(trainLoader,testLoader,epochs=5,lr=1e-3):
    model=ClassicLinear()
    acc=trainModel(model,trainLoader,testLoader,epochs=epochs,lr=lr)
    ips,ms=benchmarkInference(model,testLoader,warmupBatches=10,benchBatches=200)
    return model,acc,ips,ms

def trainBlocked(trainLoader,testLoader,epochs=5,lr=1e-3):
    model=BlockedLinearModel()
    acc=trainModel(model,trainLoader,testLoader,epochs=epochs,lr=lr)
    ips,ms=benchmarkInference(model,testLoader,warmupBatches=10,benchBatches=200)
    return model,acc,ips,ms

def main(argc:int,*argv:str)->int:
    torch.manual_seed(0)

    batchSize=256
    epochs=5
    lr=1e-3

    trainLoader,testLoader=getMnistLoaders(batchSize=batchSize,numWorkers=2)

    classicModel,classicAcc,classicIps,classicMs=trainClassic(trainLoader,testLoader,epochs=epochs,lr=lr)
    blockedModel,blockedAcc,blockedIps,blockedMs=trainBlocked(trainLoader,testLoader,epochs=epochs,lr=lr)

    plotAccuracyCurves(classicAcc,blockedAcc,outPath="mnist_accuracy.png")
    plotInferenceBars(classicIps,blockedIps,outPath="mnist_inference_speed.png")

    print(f"Final Test Acc: classic={classicAcc[-1]:.4f} blocked={blockedAcc[-1]:.4f}")
    print(f"Inference: classic={classicIps:.1f} img/s ({classicMs:.4f} ms/img)  blocked={blockedIps:.1f} img/s ({blockedMs:.4f} ms/img)")
    print("Saved plots: mnist_accuracy.png, mnist_inference_speed.png")
    return 0

if __name__=="__main__":
    argv=__import__("sys").argv
    exit(main(len(argv),*argv))
