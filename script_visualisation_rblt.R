

library(rblt)


fileh5s<-dir("D:/Martinique/Analyse/Acceleration corrigee h5", full.names=T)
print(fileh5s)
fileh5<-fileh5s[10]
print(fileh5)


#1- classic plot
l=LoggerList$new()
l$add(LoggerCats$new(fileh5, metricshow=c(T,T,F,F,T,F)))
lui=rblt:::LoggerUI$new(l)
lui$gui()


##Visualisation with predictions

fileh5s<-dir("D:/Martinique/Analyse/Acceleration corrigee h5", full.names=T)
print(fileh5s)
fileh5<-fileh5s[8]
print(fileh5)


fileBehaviors<-dir("C:/Users/loren/Documents/Tortues/Martinique_Verte/Behavior", pattern = "\\.csv$", full.names=T)
print(fileBehaviors)  
fileBehavior=fileBehaviors[8] 
print(fileBehavior)

lg=LoggerCats$new(fileh5,filebehavior=fileBehavior,besep=";",metricshow=c(T,T,F,F,T,F),besaturation=0.7)
l=LoggerList$new()
l$add(lg)
lui=rblt:::LoggerUI$new(l)
lui$gui()