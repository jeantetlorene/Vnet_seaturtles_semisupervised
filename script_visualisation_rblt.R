

library(rblt)


fileh5s<-dir("C:/Users/loren/Documents/Tortues/Martinique_Verte/fichiers_h5", full.names=T)
fileh5s<-dir("D:/Martinique/Analyse/Acceleration corrigee h5", full.names=T)
fileh5s<-dir("D:/Martinique/Leo/fichiers_h5_without_correction", full.names=T)
fileh5s<-dir("E:/Martinique/Leo/fichiers_h5_with_correction", full.names=T)
print(fileh5s)
fileh5<-fileh5s[14]
print(fileh5)


#1- classic plot
l=LoggerList$new()
l$add(LoggerCats$new(fileh5, metricshow=c(T,T,F,F,T,F)))
lui=rblt:::LoggerUI$new(l)
lui$gui()


##Visualisation with predictions

fileh5s<-dir("E:/Martinique/Leo/fichiers_h5_with_correction", full.names=T)
print(fileh5s)
fileh5<-fileh5s[14]
print(fileh5)


fileBehaviors<-dir("C:/Users/loren/Documents/Tortues/Martinique_Verte/Behavior_rblt", pattern = "\\.csv$", full.names=T)
print(fileBehaviors)  
fileBehavior=fileBehaviors[1] 
print(fileBehavior)

lg=LoggerCats$new(fileh5,filebehavior=fileBehavior,besep=";",metricshow=c(T,T,F,F,T,F),besaturation=0.7)
l=LoggerList$new()
l$add(lg)
lui=rblt:::LoggerUI$new(l)
lui$gui()