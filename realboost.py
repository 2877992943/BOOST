import random
import os
import sys
import math
import copy



inpath = "D://python2.7.6//MachineLearning//real boost//me-1"
inpath2 = "D://python2.7.6//MachineLearning//real boost//me-1"
outfile1 = "D://python2.7.6//MachineLearning//real boost//1.txt"
outfile2 = "D://python2.7.6//MachineLearning//real boost//2.txt"
outfile3 = "D://python2.7.6//MachineLearning//real boost//3.txt"
outfile4 = "D://python2.7.6//MachineLearning//real boost//4.txt"
     
clas='business'
classList=['business','it','yule','sports','auto']
eps=0.0000001
allStumps=50
######################

def loadData():
    wordDic={}
    docList=[]
    

    ###################build docList  
    
    for filename in os.listdir(inpath):
        for c in classList:
            if filename.find(c)!=-1:
                eachDoc=[{},c,0.0,0.0]
        content=open(inpath+'/'+filename,'r').read().strip()
        words=content.replace('\n',' ').split(' ')
        for word in words:
            if len(word.strip())<=2:continue
            if word not in wordDic:
                wordDic[word]=1 
                eachDoc[0][word]=1;
                
            elif word in wordDic:
                if word not in eachDoc[0].keys():
                    eachDoc[0][word]=1
                else:eachDoc[0][word]+=1
        docList.append(eachDoc)

    print '%d docs loaded and %d feature...'%(len(docList),len(wordDic))
    numSample=len(docList);    #differenc between 1/float(a)  and float(1/int(a))
    #############################sample weight 
    for doc in docList:
        doc[2]=1/float(numSample); 
        if doc[1]!=clas:
            doc[1]=-1
        else:doc[1]=1
        
    ###################
    for word in wordDic:
        max1=None;max2=None
        for doc in docList:
            if word in doc[0]:
                if doc[1]==1:
                    if max1==None or max1<doc[0][word]:
                        max1=doc[0][word]
                else:
                    if max2==None or max2<doc[0][word]:
                        max2=doc[0][word]
        wordDic[word]={1:max1,-1:max2}
                    
            
                

    ###################output    
    outPutfile=open(outfile1,'w')
    for (wid,max12) in wordDic.items():
        outPutfile.write(str(wid));
        outPutfile.write(' ')
        outPutfile.write(str(max12))
        outPutfile.write('\n')
    outPutfile.close()
 
    outPutfile=open(outfile2,'w')
    for doc in docList:
        outPutfile.write(str(doc[0]))
        outPutfile.write('\n')    
        outPutfile.write(str(doc[1]))
        outPutfile.write(' ')
        outPutfile.write(str(doc[2]))
        outPutfile.write(' ')
        outPutfile.write(str(doc[3]))
        outPutfile.write(' ')
        outPutfile.write('\n')
    outPutfile.close() 

    return wordDic,docList
    

def errCount(docList):
    #print len(judgeList),len(docList)
    err=0.0; 
    for doc in docList:
        if doc[3]==False:
            err+=doc[2]
    return err

def stump(wordDic,docList):
    allfList=[]
     
    minErr=None;
    stumpInfo=[] #weak classifier info
    for wid,max12 in wordDic.items():
        maxfreq=max(max12.values())
        for dim in range(maxfreq):
            for choice in [1,-1]:  ### '+ -'   '- +'
                 
                
                
                for doc in docList:
                    if wid not in doc[0] and dim==0:
                        doc[3]=choice
                        if doc[3]!=doc[1]:doc[3]=False#judgeList.append(doc[3]==doc[1]) ;#print '1',judgeList
                    if wid not in doc[0] and dim!=0:
                        doc[3]=choice;
                        if doc[3]!=doc[1]:doc[3]=False;#print '11',judgeList
                    if wid in doc[0] and doc[0][wid]<=dim+1:   #what if the word not in the doc??
                        doc[3]=choice
                        if doc[3]!=doc[1]:doc[3]=False;#print '2',judgeList
                    if wid in doc[0] and doc[0][wid]>dim:
                        doc[3]=-choice
                        if doc[3]!=doc[1]:doc[3]=False
                        
                
                error=errCount(docList);#print error
                if minErr==None or minErr>error:
                    minErr=error; print 'minerr', minErr  
                    stumpInfo=[wid,dim,choice];#print stumpInfo

    ############remove the feat that been used in the stump
    wordDicNew=wordDic.copy();#print 'dic1',len(wordDicN)  #184
    del wordDicNew[stumpInfo[0]];#print len(wordDic),len(wordDicN)   #184  183   

    return stumpInfo,wordDicNew

def split(stumpInfo,docList):   #split and  calc prob
    wid=stumpInfo[0]
    for doc in docList:
        if wid not in doc[0]:
            doc[3]=stumpInfo[2]
        elif wid in doc[0]:
            if doc[0][wid]<=stumpInfo[1]:
                doc[3]=stumpInfo[2]   #lable
            else:doc[3]=-stumpInfo[2]

    #######calculate p(y=1|x) for left right bag
    leftBag=0.0;rightBag=0.0;probLeft=0.0;probRight=0.0
    for doc in docList:
        if doc[3]==1:
            leftBag+=doc[2]
            if doc[1]==1:
                probLeft+=doc[2]

        if doc[3]==-1:
            rightBag+=doc[2]
            if doc[1]==1:
                probRight+=doc[2]
    #print probLeft,leftBag,probRight,rightBag            
    probLeft=probLeft/leftBag ;print 'prob', probLeft
    probRight=probRight/rightBag;print probRight
    probLeft=0.5*math.log(eps+probLeft/(eps+1-probLeft))
    probRight=0.5*math.log(eps+probRight/(eps+1-probRight))
    stumpInfo.append([probLeft,probRight]);#print stumpInfo[0],stumpInfo
    return stumpInfo,docList
        

def sigm(sumf):
    if sumf>0.0:
        return 1
    else: return -1
    
def updateWeight(docList,stumpInfo):
    #print docList
    wid=stumpInfo[0]
    for doc in docList:
        if wid not in doc[0]:
            doc[2]=doc[2]*math.exp(-doc[1]*stumpInfo[3][0])
        if wid in doc[0]:
            if doc[0][wid]<=stumpInfo[1]:
                predictLabel=stumpInfo[2]
                if predictLabel==1:
                    doc[2]=doc[2]*math.exp(-doc[1]*stumpInfo[3][0])
                else:
                    doc[2]=doc[2]*math.exp(-doc[1]*stumpInfo[3][1])
                    

            if doc[0][wid]>stumpInfo[1]:
                predictLabel=-stumpInfo[2]
                if predictLabel==1:
                    doc[2]=doc[2]*math.exp(-doc[1]*stumpInfo[3][0])
                else:
                    doc[2]=doc[2]*math.exp(-doc[1]*stumpInfo[3][1])

    #####renormalize
    totalW=0.0
    for doc in docList:
        #print doc[2]
        totalW+=doc[2]
    #print totalW
    for doc in docList:
        doc[2]/=totalW;
        #print doc[2]
    #print docList
    return docList
        



def test(strong):
    i=0;testDocList=[]
    for filename in os.listdir(inpath2):
        for c in classList:
            if filename.find(c)!=-1:
                eachDoc=[{},filename,c,'predict']
                testDocList.append(eachDoc)
        i+=1;
        if i>10:break

    sampleList=random.sample(testDocList,5);#print sampleList
    #############read doc
    for i in range(len(sampleList)):
        filename=sampleList[i][1]; 
        content=open(inpath2+'/'+filename,'r').read().strip()
        words=content.replace('\n',' ').split(' ')
        for word in words:
            if word.strip()<=2:continue
            if word not in sampleList[i][0]:
                sampleList[i][0][word]=1
            else:
                sampleList[i][0][word]+=1
        sampleList[i].remove(filename);#print sampleList[i][1:]#del sampleList[i][1]
        if sampleList[i][1]==clas:
            sampleList[i][1]=1
        else:sampleList[i][1]=-1
        print sampleList[i][1:]
     

    ##########
    outPutfile=open(outfile3,'w')
    for doc in sampleList:
        outPutfile.write(str(doc));
        outPutfile.write('\n')
    outPutfile.close()
    #########predict
    for doc in sampleList:
        sumf=0.0
        for stump in strong:
            if stump[0] not in doc[0]:
                predict=stump[2]
                if predict==1:
                    sumf+=float(stump[3][0])
                else:sumf+=float(stump[3][1])
                    

            if stump[0] in doc[0]:
                if doc[0][stump[0]]<=stump[1]:
                    predict=stump[2]
                    if predict==1:
                        sumf+=float(stump[3][0])
                    else:sumf+=float(stump[3][1])
                elif doc[0][stump[0]]>stump[1]:
                    predict=-stump[2]
                    if predict==1:
                        sumf+=float(stump[3][0])
                    else:sumf+=float(stump[3][1])
        print sumf
        doc[2]=sigm(sumf)

    err=[]
    for doc in sampleList:
        err.append(doc[1]==doc[2])
    print err
    print float(err.count(False))/float(len(sampleList))
        
                    
                
                    
                    
    
        



            
    
    
        
                
                





        
    
  
###################main
wordDic,docList=loadData()

stumpInfo,wordDicNew=stump(wordDic,docList)
stumpInfo,docList=split(stumpInfo,docList)
docList=updateWeight(docList,stumpInfo)

i=0
strongClassifier=[]
while i<allStumps:
    #del wordDicNew[stumpInfo[0]]#####
    wordDic0=wordDicNew.copy()#wordDic0=wordDicNew   #not work same feat appear
    stumpInfo,wordDicNew=stump(wordDic0,docList);#print 'len',len(wordDicNew)
    stumpInfo,docList=split(stumpInfo,docList)
    strongClassifier.append(stumpInfo);print stumpInfo[0],stumpInfo
    docList=updateWeight(docList,stumpInfo)
    i+=1



test(strongClassifier)   







    
    
