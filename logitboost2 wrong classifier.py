import random
import os
import sys
import math
import copy



inpath = "D://python2.7.6//MachineLearning//boost//me-0"
inpath2 = "D://python2.7.6//MachineLearning//boost//me-2"
outfile1 = "D://python2.7.6//MachineLearning//boost//1.txt"
outfile2 = "D://python2.7.6//MachineLearning//boost//2.txt"
outfile3 = "D://python2.7.6//MachineLearning//boost//3.txt"
outfile4 = "D://python2.7.6//MachineLearning//boost//4.txt"
     
clas='business'
classList=['business','it','yule','sports','auto']
eps=0.000001
allStumps=5
######################

def loadData():
    wordDic={}
    docList=[]
    

    ###################build docList  
    
    for filename in os.listdir(inpath):
        for c in classList:
            if filename.find(c)!=-1:
                eachDoc=[{},c,'truelabel',['predict','z','z'],[],'F']
        content=open(inpath+'/'+filename,'r').read().strip()
        words=content.replace('\n',' ').split(' ')
        for word in words:
            if len(word.strip())<=2:continue
            if word not in wordDic:
                wordDic[word]=0.0  #calculate entropy need to 0- from 0 
                eachDoc[0][word]=1;
            elif word in wordDic:
                #wordDic[word]+=1
                if word not in eachDoc[0].keys():
                    eachDoc[0][word]=1
                else:eachDoc[0][word]+=1
        docList.append(eachDoc)

    print '%d docs loaded and %d feature...'%(len(docList),len(wordDic))
    numSample=float(len(docList));    #differenc between 1/float(a)  and float(1/int(a))
    #######entropy of wid
    wordDic=minEntropy(wordDic,docList)
    wordDic=halfwid(wordDic)

    #############################sample weight ,true label
    for doc in docList:
        doc[4]=1/float(numSample); 
        if doc[1]!=clas:
            doc[2]=-1
        else:doc[2]=1
  

    ###################output    
    outPutfile=open(outfile1,'w')
    for (wid,freq) in wordDic.items():
        outPutfile.write(str(wid));
        outPutfile.write(' ')
        outPutfile.write(str(freq))
        outPutfile.write('\n')
    outPutfile.close()
 
    outPutfile=open(outfile2,'w')
    for doc in docList:
        outPutfile.write(str(doc[0]))
        outPutfile.write('\n')    
        outPutfile.write(str(doc[1:]))
        outPutfile.write('\n')
    outPutfile.close() 

    return wordDic,docList

#################################################################################
#################################################################################
def minEntropy(wordDic,docList):
    for wid,v in wordDic.items():
        numDocEachC=[0.0,0.0,0.0,0.0,0.0] #for some wid,one class like business,how many doc contain wid,

        for i in range(len(classList)):
            for doc in docList:
                if doc[1]==classList[i] and wid in doc[0]:
                    numDocEachC[i]+=1
        #print '1', numDocEachC
        allDoc=float(sum(numDocEachC))#fenmu, for all doc ,how many is there contain the wid
        numDocEachC=[i/allDoc for i in numDocEachC];#print 'numDock',numDock
        #print '2',numDocEachC
        
        for i in range(len(classList)):
            if numDocEachC[i]>0.0:
                wordDic[wid]-=math.log(numDocEachC[i])*numDocEachC[i]  #from 0 minus

    return wordDic

def halfwid(wordDic):
    print len(wordDic)
    midEnt=(max(wordDic.values())-min(wordDic.values()))/100.0
    for w,f in wordDic.items():
        if f>=midEnt+min(wordDic.values()):
            del wordDic[w]

    print len(wordDic)
    return wordDic
    

def calcZW(docList,wordDic,stumpInfo):
    for doc in docList:
        #print doc[1:]
        yixing=(doc[2]+1)/2 ;#print 'yi*',yixing
        
        z1=(yixing-stumpInfo[1][0])/(eps+stumpInfo[1][0]*(1-stumpInfo[1][0]))
        z2=(yixing-stumpInfo[1][1])/(eps+stumpInfo[1][1]*(1-stumpInfo[1][1]))
        doc[3]=['predict',z1,z2];#print 'z',doc[3]
        w1=stumpInfo[1][0]*(1-stumpInfo[1][0])
        w2=stumpInfo[1][1]*(1-stumpInfo[1][1])
        doc[4]=[w1,w2];#print 'w',doc[4]

    return docList
    
def fitting(docList,wordDic,stumpInfo):
    #########get best wid mark
    minErr=None
    for wid in wordDic.keys():
        for mark in [1,-1]:
            ls=0.0
            for doc in docList:
                ##########
                if wid in doc[0]:
                    predict=mark
                else:predict=-mark
                doc[3][0]=predict
                ##########
                bag=(1-predict)/2
                z0=(0-stumpInfo[1][bag])/(eps+stumpInfo[1][bag]*(1-stumpInfo[1][bag]));#print 'z0',z0#use Pleft to calculate 
                z1=(1-stumpInfo[1][bag])/(eps+stumpInfo[1][bag]*(1-stumpInfo[1][bag]));#print 'z1',z1#use Pleft
                stumpInfo[2][bag]=z1*stumpInfo[1][bag]+z0*(1-stumpInfo[1][bag]) ;#print'fm',stumpInfo[2][bag] #get fm left
                lsi=(doc[3][bag+1]-stumpInfo[2][bag])**2  ;#print 'xi ls',ls#least sqaure
                ls+=lsi*doc[4][bag];#print 'weighted least square',ls

            if minErr==None or minErr>ls:
                minErr=ls
                bestFeat=wid
                if bag==0:mark=1
                elif bag==1:mark=-1
                stumpInfo[0]=[bestFeat,mark];#print bestFeat,stumpInfo  #fm=Ew(z|x)output 0
    print stumpInfo[0][0]
    ##########use this wid mark to split xi
    for doc in docList:
        wid=stumpInfo[0][0];mark=stumpInfo[0][1]
        if wid in doc[0]:
            doc[3][0]=mark
        else:doc[3][0]=-mark
        #print 'true,predict',doc[2],doc[3][0]

    del wordDic[stumpInfo[0][0]]

    return stumpInfo,wordDic,docList

def calcPFfm(docList,stumpInfo):
    feat=stumpInfo[0][0]
    mark=stumpInfo[0][1] #1  or -1
    for doc in docList:
        #####mark the predict label
        if feat in doc[0]:
            doc[3][0]=mark
        else:doc[3][0]=-mark
        predict=doc[3][0]
        ####predict 1,-1 correspond to stumpInfo 0 ,1
        #if predict==1:bag=0
        #if predict==-1:bag=1
        bag=(1-predict)/2
        #######z0 z1 F1 F2   pi(y=1|xi)
        z0=(0-stumpInfo[1][bag])/(eps+stumpInfo[1][bag]*(1-stumpInfo[1][bag]));#print 'z0',z0,
        z1=(1-stumpInfo[1][bag])/(eps+stumpInfo[1][bag]*(1-stumpInfo[1][bag]));#print 'z1',z1
        i=0
        F1=doc[5][0];
        initialF2=1
        F2=F1+0.5*(z0*(1-1/(1+math.exp(-2*initialF2)))+z1*(1/(1+math.exp(-2*initialF2))))
        while abs(F2-initialF2)>eps and i<10:
            i+=1;#print 'iter',i
            initialF2=F2;#print 'F2',initialF2
            F2=F1+0.5*(z0*(1-1/(1+math.exp(-2*initialF2)))+z1*(1/(1+math.exp(-2*initialF2))));
        #####each xi , save pi,F,fm
        pi=1/(1+math.exp(-2*F2))
        doc[5][0]+=F2;doc[5][1]=pi;
        ####fm two ways output same fm
        fm=F2-F1; 
        #fm1=z0*(1-pi)+z1*pi;print fm
        doc[5][2]=fm;print 'F,p,fm',doc[5];print doc[2],doc[3][0]
    ########## weighted  fm  and  P(y=1)  for left and right branch, to be saved in stumpInfo
    for c in [1,-1]:
        wtotal=0.0
        fm=0.0;p=0.0
        bag=(1-c)/2 #1,-1 correspond to 0,1
        for doc in docList:
            if doc[3][0]==c:
                wtotal+=doc[4][bag]
        for doc in docList:
            if doc[3][0]==c:
                fm+=doc[5][2]*doc[4][bag]
                p+=doc[5][1]*doc[4][bag]
        fm/=wtotal;p/=wtotal
        stumpInfo[2][bag]=fm;stumpInfo[1][bag]=p

    return stumpInfo,docList

        
        
                
            
                

    


def sigm(sumf):######to the contrary
    if sumf>0.0:
        return -1
    else: return 1
#################################################################################
##################################################################################
def stump(wordDic,docList):
    ########initial F(xi),p(y=1|x)left and right
    for doc in docList:
        doc[5]=[0.0,0.0,0.0]#[F,pi,fm] all need calculated with weighted
    #stumpInfo=[['feat','featvalue'],'pleft','pright','fmleft','fmriht'] #weak classifier info
    stumpInfo=[['feat','featvalue'],[0.5,0.5],[0.0,0.0]]
    ##################### for stump=1:m
    i=1
    while i<=10:  #for stump 1:m
        #calc w z in docList
        docList=calcZW(docList,wordDic,stumpInfo)
        #get (wid mark) in stumpInfo,fitting least-square,del wid in wordDic
        stumpInfo,wordDic,docList=fitting(docList,wordDic,stumpInfo) 
        #calc F for each xi in docList,get p(y=1) in stumpInfo
        stumpInfo,docList=calcPFfm(docList,stumpInfo)
        i+=1
 
        


    

 
        



def test(strong,wordDic):
    ##########sample testdoc
    i=0;testDocList=[]
    for filename in os.listdir(inpath2):
        for c in classList:
            if filename.find(c)!=-1:
                eachDoc=[{},filename,c,'predict','weight']
                testDocList.append(eachDoc)
        i+=1;
        if i>200:break

    sampleList=random.sample(testDocList,100);#print sampleList
    #############read doc
    for i in range(len(sampleList)):
        filename=sampleList[i][1]; 
        content=open(inpath2+'/'+filename,'r').read().strip()
        words=content.replace('\n',' ').split(' ')
        for word in words:
            if word.strip()<=2:continue
            #if word not in wordDic:
                #wordDic[word]=1
            if word not in sampleList[i][0]:
                sampleList[i][0][word]=1
            elif word in sampleList[i][0]:
                sampleList[i][0][word]+=1
        sampleList[i].remove(filename);#print sampleList[i][1:]#del sampleList[i][1]
        if sampleList[i][1]==clas:
            sampleList[i][1]=1
        else:sampleList[i][1]=-1
        #print sampleList[i][1:]
     

    ##########
    outPutfile=open(outfile3,'w')
    for doc in sampleList:
        outPutfile.write(str(doc));
        outPutfile.write('\n')
    outPutfile.close()
    #########predict
    for doc in sampleList:
        #print 'doc',sampleList.index(doc)
        sumf=0.0
        #####each stump vote
        for stump in strong:
            [wid,mark]=stump[0]
            fm=0.0
            if wid in doc[0]:
                predict=mark
                if predict==1:
                    fm=stump[3]
                else:fm=stump[4]

            elif wid not in doc[0]:
                predict=-mark
                if predict==1:
                    fm=stump[3]
                else:fm=stump[4]
            #print 'weak fm',strong.index(stump),fm
            sumf+=fm
             
        #print 'true label',doc[1]    
        #print 'sum f classifier weak ',sumf
        doc[2]=sigm(sumf)

    err=[]
    for doc in sampleList:
        err.append(doc[1]==doc[2])
    print err
    print float(err.count(False))/float(len(sampleList))
        
  
    
  
###################main
wordDic,docList=loadData()



stump(wordDic,docList)
 
'''docList=updateWeight(docList,stumpInfo)


i=0
strongClassifier=[]
while i<=allStumps:
    print 'stump No.',i
    wid=stumpInfo[0][0];del wordDicNew[wid]#####
    wordDic0=wordDicNew.copy()#wordDic0=wordDicNew   #not work same feat appear
    stumpInfo,wordDicNew,docList=stump(wordDic0,docList);#print 'len',len(wordDicNew)
    strongClassifier.append(stumpInfo);print stumpInfo[0][0],stumpInfo
    docList=updateWeight(docList,stumpInfo)
    i+=1



test(strongClassifier,wordDic)'''  






    
    
