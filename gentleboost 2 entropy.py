import random
import os
import sys
import math
import copy



inpath = "D://python2.7.6//MachineLearning//boost//me-1"
inpath2 = "D://python2.7.6//MachineLearning//boost//me-2"
outfile1 = "D://python2.7.6//MachineLearning//boost//1.txt"
outfile2 = "D://python2.7.6//MachineLearning//boost//2.txt"
outfile3 = "D://python2.7.6//MachineLearning//boost//3.txt"
outfile4 = "D://python2.7.6//MachineLearning//boost//4.txt"
     
clas='business'
classList=['business','it','yule','sports','auto']
eps=0.000001
allStumps=10
######################

def loadData():
    wordDic={}
    docList=[]
    

    ###################build docList  
    
    for filename in os.listdir(inpath):
        for c in classList:
            if filename.find(c)!=-1:
                eachDoc=[{},c,'truelabel','predictlabel',0.0]
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
    

def errCount(docList):
    #print len(judgeList),len(docList)
    err=0.0; 
    for doc in docList:
        if doc[2]!=doc[3]:
            err+=doc[4]###err+=1 not right add weight
    return err

def calcProb(stumpInfo,docList):
    #####split
    [wid,mark]=stumpInfo[0] 
    for doc in docList:
        if wid in doc[0].keys():
            doc[2]=mark
        else:doc[2]=-mark
    ########
    #truePos=0.0;trueNeg=0.0   wrong setting
    #prePos=0.0;preNeg=0.0

    prePos=0.0;preNeg=0.0;fenzi1=0.0;fenzi2=0.0
    for doc in docList:
        if doc[3]==1:
            prePos+=doc[4]   #fen mu
        if doc[3]==-1:
            preNeg+=doc[4]   #fen mu
        if doc[3]==1 and doc[2]==1:
            fenzi1+=doc[4]
        if doc[3]==-1 and doc[2]==1:
            fenzi2+=doc[4]
    #########
    p1L=fenzi1/float(prePos+eps);stumpInfo[1]=p1L
    p1R=fenzi2/float(preNeg+eps);stumpInfo[2]=p1R
    stumpInfo[3]=2*p1L-1
    stumpInfo[4]=2*p1R-1
    print stumpInfo
    return stumpInfo,docList

def stump(wordDic,docList):
    allfList=[]
     
    minErr=None;err=0.0
    stumpInfo=[['feat','featvalue'],'pleft','pright','fmleft','fmriht'] #weak classifier info
    for wid,freq in wordDic.items():
        for mark in [1,-1]:

            for doc in docList:
                if wid in doc[0].keys():
                    doc[3]=mark    #predict: doc[2]
                    featValue=mark;#print wid,featValue  
                    
                elif wid not in doc[0].keys():
                    doc[3]=-mark
                    featValue=-mark;#print wid,featValue

            err=errCount(docList);#print 'err', err,minErr
            if minErr==None or minErr>err:
                #print 'featValue',featValue,minErr,wid#####
                minErr=err
                stumpInfo[0]=[wid,featValue]

    #print 'final feat',stumpInfo[0][0],stumpInfo[0][1]
    ###########already find best feat by counting err,now count prob and fm output
    stumpInfo,docList=calcProb(stumpInfo,docList) ###docList updated at the predict position        
 
    ############remove the feat that been used in the stump
    wordDicNew=wordDic.copy();#print 'dic1',len(wordDicN)  #184
    widDel=stumpInfo[0][0]
    #del wordDicNew[widDel];#print len(wordDic),len(wordDicN)   #184  183   

    return stumpInfo,wordDicNew,docList

 
        

def sigm(sumf):
    if sumf>0.0:
        return 1
    else: return -1
    
def updateWeight(docList,stumpInfo):
   
    [wid,mark]=stumpInfo[0] 
    for doc in docList:
        if doc[3]==1:
            #print 'change',doc[1],doc[2],stumpInfo[3],doc[3]
            doc[4]=doc[4]*math.exp(-float(doc[2])*stumpInfo[3]);
            #print doc[3]
        if doc[3]==-1:
            #print 'change',doc[1],doc[2],stumpInfo[4],doc[3]
            doc[4]=doc[4]*math.exp(-float(doc[2])*stumpInfo[4]);
            #print doc[3]

    #####renormalize
    totalW=0.0
    for doc in docList:
        totalW+=doc[4]
    for doc in docList:
        doc[4]/=totalW;
        #print doc[3]
     
    return docList
        



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


stumpInfo,wordDicNew,docList=stump(wordDic,docList)
 
docList=updateWeight(docList,stumpInfo)


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



test(strongClassifier,wordDic)  






    
    
