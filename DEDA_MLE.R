# ------------------------------------------------------------------------------
# Published in: Digital Economy and Decision Analytics
# ------------------------------------------------------------------------------
# Description:  The following code uses preprocessed data on Berlin housing
#               market. LASSO regression is conducted on 10 folds of data and    
#               the in each fold selected variables are presented in the plot.
#               The code is based on the code presented in Mullainathan, 
#               Sendhil, and Jann Spiess. 2017. "Machine Learning: An Applied 
#               Econometric Approach." Journal of Economic Perspectives, 
#               31(2): 87-106.
# ------------------------------------------------------------------------------
# Keywords:     MLE, Machine Learning, Econometrics, LASSO 
# ------------------------------------------------------------------------------
# Author:       Karolina Stanczak
# ------------------------------------------------------------------------------
# Submitted:    2017/07/24
# ------------------------------------------------------------------------------
# Datafile:     dataprepall.rdata
# ------------------------------------------------------------------------------
# Input:        -
# ------------------------------------------------------------------------------
# Output:       Plot presenting the loss function of LASSO regression. Plot 
#               showing selected variables in each of ten folds that the data  
#               is divided into.
# ------------------------------------------------------------------------------


# Function for installing & loading required packages.
loadinstallpackages = function(packs){
  new.packs = packs[!(packs %in% installed.packages()[, "Package"])]
  if (length(new.packs)) 
    install.packages(new.packs, dependencies = TRUE)
  sapply(packs, require, character.only = TRUE)
}

# Packages we will need for our code are inside "packs"
packs = cbind("glmnet", "reshape2", "ggplot2")
loadinstallpackages(packs)

# Load the preprocessed Berlin Housing Data 
basedata = readRDS(file=paste0("dataprepall.rdata"))
localahs = basedata$df
Y        = localahs[,"LOGKP"]
thisrhs  = paste(basedata$vars,collapse=" + ")

# Prepare the model formula
X = model.matrix(as.formula(paste("LOGKP", thisrhs, sep = " ~ ")),localahs)

# Select (and fix) tuning parameter
set.seed(123)
firstsubsample = sample(nrow(localahs),nrow(localahs)/10)

firstlasso = glmnet(X[firstsubsample,],Y[firstsubsample])
losses     = apply(predict(firstlasso,newx=X[-firstsubsample,]),2,
                   function(Yhat) mean((Yhat - Y[-firstsubsample])^2))

# Plot the loss function
plot(log(firstlasso$lambda),losses)

lambda = firstlasso$lambda[which.min(losses)]

# Fit LASSO models
I = length(unique(localahs$lassofolds))

barcodes       = matrix(0,nrow=I,ncol=firstlasso$dim[1])
lassonormcoeff = matrix(0,nrow=I,ncol=firstlasso$dim[1])
lassolosses    = vector(mode='numeric',length=I)
lassose        = vector(mode='numeric',length=I)

# Calculate the models
for(i in 1:I) {
  thissubsample = localahs$lassofolds == i
  
  thislasso  = glmnet(X[thissubsample,],Y[thissubsample])
  thislosses = apply(predict(thislasso,newx=X[-thissubsample,]),2
                     ,function(Yhat) mean((Yhat - Y[-thissubsample])^2))
  thislambda = firstlasso$lambda[which.min(thislosses)]
  
  barcodes[i,as.vector(!(thislasso$beta[,which.min(thislosses)] == 0))] = 1
  pointlosses    = (predict(thislasso,newx=X[-thissubsample,],s=thislambda) - 
                   Y[-thissubsample])^2
  lassolosses[i] = mean(pointlosses)
  lassose[i]     = sd(pointlosses) / sqrt(length(pointlosses))
}


# Barcode plot
barcodeplotdata           = melt(barcodes)
names(barcodeplotdata)    = c("Iteration","Coefficient","Selected")
barcodeplotdata$Selected  = as.factor(barcodeplotdata$Selected)
barcodeplotdata$Iteration = as.factor(barcodeplotdata$Iteration)

barcodeplot = ggplot(data = barcodeplotdata, aes(x=Iteration, y=Coefficient, 
                                                 fill=Selected)) +  
  geom_tile() + scale_fill_manual(values=c("white","black"),
                                  labels=c("zero", "nonzero")) + theme_bw() +
  labs(list(x = "Fold of the sample", y = "Parameter in the linear model",
            fill="Estimate")) + theme(axis.text.y=element_blank(),
        axis.ticks.y=element_blank())

print(barcodeplot)
ggsave(barcodeplot,file="barcode.png")