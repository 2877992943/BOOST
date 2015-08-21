import random
import os
import sys
import math
import copy



inpath = "D://python2.7.6//MachineLearning//boost//me-2"
inpath2 = "D://python2.7.6//MachineLearning//boost//me-all"
outfile1 = "D://python2.7.6//MachineLearning//boost//1.txt"
outfile2 = "D://python2.7.6//MachineLearning//boost//2.txt"
outfile3 = "D://python2.7.6//MachineLearning//boost//3.txt"
outfile4 = "D://python2.7.6//MachineLearning//boost//4.txt"
     
clas='yule'
classList=['business','it','yule','sports','auto']
eps=0.000001
allStumps=15
######################

def loadData():
    wordDic={}
    docList=[]
    

    ###################build docList  
    
    for filename in os.listdir(inpath):
        for c in classList:
            if filename.find(c)!=-1:
                eachDoc=[{},c,'truelabel','predictlabel','pi','wi','F']
        content=open(inpath+'/'+filename,'r').read().strip()
        words=content.replace('\n',' ').split(' ')
        for word in words:
            if len(word.strip())<=2:continue
            if word not in wordDic:
                wordDic[word]=0.0 #calc entropy minus from 0
                eachDoc[0][word]=1;
                
            elif word in wordDic:
                if word not in eachDoc[0].keys():
                    eachDoc[0][word]=1
                else:eachDoc[0][word]+=1
        docList.append(eachDoc)

    print '%d docs loaded and %d feature...'%(len(docList),len(wordDic))
    numSample=float(len(docList));    #differenc between 1/float(a)  and float(1/int(a))
    #######entropy of wid
    wordDic=minEntropy(wordDic,docList)
    wordDic=halfwid(wordDic)


    #############################F,wi,pi,
    for doc in docList:
        doc[5]=1/float(numSample); #wi
        if doc[1]!=clas:
            doc[2]=[-1,0,'zi']   #y y* zi
        else:doc[2]=[1,1,'zi']
        doc[6]=0.0   #F
        doc[4]=0.5   #pi
        
  

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
######################################################################
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
    print 'feat',len(wordDic)
    midEnt=(max(wordDic.values())-min(wordDic.values()))/200.0
    for w,f in wordDic.items():
        if f>=midEnt+min(wordDic.values()):
            del wordDic[w]

    print 'feat',len(wordDic)
    return wordDic

 
def errCount(docList):
    err=0.0
    for doc in docList:
        ####(zi-yi)^2*wi sum
        err+=((doc[3]-doc[2][2])**2)*doc[5]
    return err

def sigm(sumf):
    if sumf>0.0:
        return 1
    else: return -1       
        
 
######################################################################
def stump(wordDic,docList):
    ######calc zi,wi
    for doc in docList:
        wi=doc[4]*(1-doc[4])
        zi=(doc[2][1]-doc[4])/(wi+eps)
        doc[5]=wi
        doc[2][2]=zi;#print 'zi y',zi,doc[2][0]
    #####################fitting Least square,find best (wid,mark)
    minErr=None
    for wid in wordDic:
        for mark in [1,-1]:
            ######calc err
            for doc in docList:
                if wid in doc[0]:
                    doc[3]=mark
                else:doc[3]=-mark
            err=errCount(docList);#print 'err',err
            ###compare err
            if minErr==None or minErr>err:
                minErr=err
                stumpInfo=[[wid,mark],{},{}]

    ##############use this wid to split xi and see whether can classify correctly
    wid=stumpInfo[0][0];mark=stumpInfo[0][1]
    for doc in docList:
        if wid in doc[0]:
            doc[3]=mark
        else:doc[3]=-mark
        #print 'predict,y,y*,zi:',doc[3],doc[2],wid
        #if doc[3]!=doc[2][0]:print 'err'  #weak classifier err rate
    ######del feat in wordDic
    del wordDic[wid]
    ##############calc fm each branch,update F in doc,calc p(y=1|xi)
    ###calc p(y=1) for each branch
    for branch in [1,-1]: #for each side of branch
        fenmu=0.0;py1=0.0
        for doc in docList:
            if doc[3]==branch:
                fenmu+=doc[5]
        for doc in docList:
            if doc[3]==branch and doc[2][0]==1: #p(y=1|x)
                py1+=doc[5]
        stumpInfo[1][branch]=py1/fenmu
    #print 'stumpInfo p',stumpInfo
    ###calc fm
    for branch in [1,-1]:
        fm=0.0;wside=0.0
        for doc in docList:
            if doc[3]==branch and doc[2][0]==1:
                fm+=doc[5]*doc[2][2]*stumpInfo[1][branch] #wi*zi*p(y=1,branch left)~wi*yi*p(y=1)
                wside+=doc[5]
            if doc[3]==branch and doc[2][0]==-1:
                fm+=doc[5]*doc[2][2]*(1-stumpInfo[1][branch])
                wside+=doc[5]
        fm/=wside
        stumpInfo[2][branch]=fm
    print 'stumpInfo',stumpInfo[0][0],stumpInfo[1:]
    #####update F for each xi
    for doc in docList:
        predict=doc[3]#go which branch
        doc[6]+=stumpInfo[2][predict]/2.0
    ####update xi
    for doc in docList:
        doc[4]=1/(1+math.exp(-2*doc[6]))
        #print 'doc',doc[1:]

    return wordDic,docList,stumpInfo

 
def test(strong,wordDic):
    ##########sample testdoc
    i=0;testDocList=[]
    for filename in os.listdir(inpath2):
        for c in classList:
            if filename.find(c)!=-1:
                eachDoc=[{},filename,c,[0,0,0],'predict','p','w','F']
                testDocList.append(eachDoc)
        i+=1;
        if i>3000:break

    sampleList=random.sample(testDocList,2000);#print sampleList
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
            sampleList[i][2][0]=1
        else:sampleList[i][2][0]=-1
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
            if wid in doc[0]:
                predict=mark
            elif wid not in doc[0]:
                predict=-mark
            fm=stump[2][predict]
                
            #print 'weak fm',strong.index(stump),fm
            sumf+=fm
             
        #print 'true label',doc[1]    
        #print 'sum f classifier weak ',sumf
        doc[3]=sigm(sumf)

    err=[]
    for doc in sampleList:
        err.append(doc[3]==doc[2][0])
    #print err
    print 'err rate',float(err.count(False))/float(len(sampleList))
        
  
    
  
###################main
wordDic,docList=loadData()

wordDic,docList,stumpInfo=stump(wordDic,docList)
 
 


i=0
strongClassifier=[]
while i<=allStumps:
    print 'stump No.',i
    wordDic,docList,stumpInfo=stump(wordDic,docList) 
    strongClassifier.append(stumpInfo)
    i+=1


test(strongClassifier,wordDic)







    
    
