# this script implements anomaly detection in time series of remote sensing data based on Isolation Forest Algorithm

library(raster)
library(rgdal)
library(caret)
library(h2o)
library(randomForest)
library(rJava)
library(ggplot2)
library(EBImage)
library(xlsx)

# Define Number of Cores of Your Cpu (including logical threads) and system memory
h2o.init(nthreads=20, max_mem_size="60G")
h2o.removeAll()

# Define path to a working directory in your system
vstupni_adresar <- choose.dir(getwd(), "Set path to a working directory")
prac_adresar <- setwd(vstupni_adresar)

# Load proper time series of remote sensing data in any of gdal library format
r <- brick(choose.files(caption = "Load Time Series for Anomaly Detection"))

# Import .csv file with time acqusition of original images properly ordered by time
sk <- read.csv2(choose.files(caption="Load .csv files with Acquisition Dates"))
vk <- as.vector(sk$x)
dk <- as.Date(vk)

# Rename raster bands based on time series data
names(r) <- c(dk)

# Beginning of time required calculations
start <- Sys.time()

raster_b <- rasterToPoints(r, progress="text")
souradnice <- as.data.frame(raster_b[,1:2])
data <- as.data.frame(raster_b[,3:nlayers(r)])
grid <- as.h2o(data)

# Own anomaly detection based on Isolation Forest Algorithm
isoforest <- h2o.isolationForest(x=colnames(grid), training_frame=grid,  ntrees=500)

# Anomalies Prediction
anomalies <- as.data.frame(predict(isoforest, grid))
anomalie <- cbind(souradnice, anomalies$predict)
R <- rasterFromXYZ(anomalie, crs=proj4string(r))
plot(R)

# Save results of detected anomalies into hard drive in Erdas Imagine raster format
exp <-  writeRaster(R, filename="Anomalies_IsoForest_H2O.img", format='HFA', overwrite = TRUE)

############################################################################################################################
############################################################################################################################

# Calculate Anomalies for every raster (each date acquisition of satellite imagery) in time series
for (i in 1:nlayers(r)){

# Load each raster band in time series raster stack
kanal <- r[[i]]

# convert each raster band into spatial point data frame
r1 <- rasterToPoints(kanal, progress="text")
df <- as.data.frame(r1)
grid1 <- as.h2o(df)

# Isolation forest model calculation (anomalies detection) for each raster band
iforest2 <- h2o.isolationForest(x=colnames(grid1), training_frame=grid1,  ntrees=500)

# Anomalies model prediction for single raster band in time series raster stack
anomalies2 <- as.data.frame(predict(iforest2, grid1))
anomalie2 <- cbind(souradnice, anomalies2$predict)
R1 <- rasterFromXYZ(anomalie2, crs=proj4string(r))

# Define a name for single raster band in time series raster stack
jmeno <- paste("anomalies_if", i, ".img", sep="_")

# Export results into hard drive for single raster band in time series raster stack 
exp <-  writeRaster(R1, filename=jmeno, format='HFA', overwrite = TRUE)
}
##############################################################################################################################
##############################################################################################################################

# Create Random Forest Regression Model between Global Anomaly Raster and Local Anomaly rasters in time t

# Create raster stack from global anomaly raster and local anomaly rasters in time t
IF_Files <- list.files(path=getwd(), pattern=".img$", full.names=T)
IF_Stack <- stack(IF_Files)

# Rename the raster bands
names(IF_Stack) <- c(paste0("B",1:nlayers(IF_Stack), col=""))

# Define number of iterations for linear model calculations
Pocet_iteraci <- 30

# The beginning of for loop for linear regression model calculation
for (i in 1:Pocet_iteraci){
  cat("Computing linear model regression", i)

# Random Sample of 2000 pixel from raster stack created in previous steps
IF_Samples <- sampleRandom(IF_Stack, size=2000, sp=T)
IF_Values <-  extract(IF_Stack, IF_Samples, df=T)

# Splitting sampled pixel for training and validation datasets....global anomaly raster...the first band in anomaly raster stack
Samples <- createDataPartition(IF_Values$B1, p = .5)[[1]]
Train <- IF_Values[Samples,]
Test <- IF_Values[-Samples,]

# Random Forest Regression Calculation
Rf_reg <- randomForest(x=Train[,-(1:2)], y=as.numeric(Train$B1),
                       ntree=1000, proximity=TRUE, importance=TRUE, confusion=TRUE, do.trace=TRUE, err.rate=TRUE,
                       mtry=sqrt(nlayers(IF_Stack)))


# Random Forest regression model prediction
p <- predict(Rf_reg, Test)

# RMSE (Root Mean Square Error) caslculation
RMSE <- sqrt(mean(((p-Test$B2))^2))

# Save results of calculated RMSE into hard drive
RMSE_Name <- paste("RMSE_", i, ".txt")
RMSE_Zaznam <- capture.output(RMSE)
RMSE_Save <- cat(RMSE_Zaznam, file=RMSE_Name, sep="\n", append=FALSE)

# Save importance of each raster bands in anomaly raster stack
tabulka_prediktory <- paste("Anomaly_importance", i, ".csv")
prediktory <- importance(Rf_reg)
export_prediktoru <- write.csv(prediktory, file=tabulka_prediktory)

# Export Variable Importance Plots into hard drive
jmeno_promenne <- paste("Predictors_RF_Regression", i, ".jpeg")
export_promennych <- jpeg(filename=jmeno_promenne, units="px", width=10000, height=10000, res=600)
varImpPlot(Rf_reg, main ="Anomaly importance")
dev.off()
}
  
###################################################################################################################
###################################################################################################################

# Visualization of MDA and MDG Importance Plots based on Random Forest Regression Model

# Create standalone directory for MDA and MDG Plots
output_dir <- file.path(getwd(), "Importance_Plots")

if (!dir.exists(output_dir)){
  dir.create(output_dir)
} else {
  print("Dir already exists!")
}

# Move files in .csv format into standalone directory created in previous step called "Importance Plots"
move <- list.files(path=getwd(), pattern="Anomaly", full.names=TRUE)
move_folder <- file.copy(from=move, to=output_dir)

# Folder called "Importance Plots is now set as new working directory
new_wd <- setwd(output_dir)

## MDA PLOT
# combine all .csv files together for MDA plot creation
csv_list <- list.files(path=new_wd, pattern="*.csv", full.names=TRUE)
csv_files_df <- lapply(csv_list, function(x){read.csv(file=x, header=TRUE, sep=",")[,2]})
csv_combined <- do.call("cbind", lapply(csv_files_df, as.data.frame))

# Label of x axis
radky <- read.csv(list.files()[c(1)])
radky2 <- as.character(radky[,1])

# Merge .csv files together and export it into gard drive as one file in .xlsx format (Microsoft Excel)
vysledky_kl <- write.xlsx(csv_combined, file="Average_Anomaly_Importance_MDA.xlsx")

# Calculation of Average Importance for each anomaly raster based on MDA
vi <- rowMeans(csv_combined)
vi2 <- cbind(csv_combined, vi)
rownames(vi2) <- c(radky2)

# MDA mean
M_MDA <- vi

# Plot Results of Average Anomaly Importance based on MDA with help of ggplot library
d <- as.data.frame(cbind(rownames(vi2),vi2$vi))
colnames(d) <- c("MDA", "Value")
d$Value <- as.numeric(as.character(d$Value))
f1 <- d[order(-d$Value),]

ggplot(f1, mapping= aes(x = reorder(MDA, Value), y = Value)) + theme_bw() +
  geom_bar(stat="identity", fill = "#FF6666") + coord_flip() + xlab(NULL) + theme_grey(base_size = 22)+
  ggtitle("Mean Decrease Accuracy")

# Export MDA plot into hard drive
png(filename="RF_MDA_GGPLOT.png", units="px", width=7000, height=5000, res=600)

ggplot(f1, mapping= aes(x = reorder(MDA, Value), y = Value)) + theme_bw() +
  geom_bar(stat="identity", fill = "#FF6666") + coord_flip() + xlab(NULL) + theme_grey(base_size = 22)+
  ggtitle("Mean Decrease Accuracy")

dev.off()
###########################################################################################################################

## MDG PLOT

# Combine all Anomaly importances in single .csv file
csv_list <- list.files(path=getwd(), pattern="*.csv", full.names=TRUE)
csv_files_df <- lapply(csv_list, function(x){read.csv(file=x, header=TRUE, sep=",")[,"IncNodePurity"]})
csv_combined <- do.call("cbind", lapply(csv_files_df, as.data.frame))

# Label of x axis
radky <- read.csv(list.files()[c(1)])
radky2 <- as.character(radky[,1])

# Merge .csv files together and export it into gard drive as one file in .xlsx format (Microsoft Excel)
vysledky_kl <- write.xlsx(csv_combined, file="Average_Anomaly_Importance_MDG.xlsx")

# Average Anomaly Importances calculation MDG
vi <- rowMeans(csv_combined)
vi2 <- cbind(csv_combined, vi)
rownames(vi2) <- c(radky2)

# MDG mean
M_MDG <- vi

# ggplot verze
d <- as.data.frame(cbind(rownames(vi2),vi2$vi))
colnames(d) <- c("MDA", "Value")
d$Value <- as.numeric(as.character(d$Value))
f2 <- d[order(-d$Value),]

ggplot(f2, mapping= aes(x = reorder(MDA, Value), y = Value)) + theme_bw() +
  geom_bar(stat="identity", fill = "#FF6666") + coord_flip() + xlab(NULL) + theme_grey(base_size = 22)+
  ggtitle("Mean Decrease Gini")

# export vizualizace na pevny disk
png(filename="RF_MDG_GGPLOT.png", units="px", width=7000, height=5000, res=600)

ggplot(f2, mapping= aes(x = reorder(MDA, Value), y = Value)) + theme_bw() +
  geom_bar(stat="identity", fill = "#00abff") + coord_flip() + xlab(NULL) + theme_grey(base_size = 22)+
  ggtitle("Mean Decrease Gini")

dev.off()

################################################################################################################################

#################################################################################################################################

# Load Exported MDA and MDG Plots

a <- readImage(files="RF_MDA_GGPLOT.png")
b <- readImage(files="RF_MDG_GGPLOT.png")

# Merge exported plots into one file
c <- abind(a, b, along=1)
display(c)

# Export merged image into hdd
d1 <- writeImage(c, files="MDA_a_MDG_together.jpeg")

#################################################################################################################################

# Correlation between MDA a MDG anomalies

# data Frame for MDA a MDG together and scaling their values between 0 a 1
dat <- as.data.frame(cbind(f1$Value, f2$Value))
colnames(dat) <- c("MDA", "MDG")
f <- function(x){(x-min(x))/(max(x)-min(x))}
dat2 <- f(dat)

# Linear relationship plot between MDA a MDG
g <- plot(dat2, main="Relationship between MDA and MDG (Scaled)", xlab="MDA", ylab="MDG", col="firebrick",
          las = 1, cex = 1.5, bty = "l", pch=16)
abline(lm(MDG~MDA, data=dat2), col="black")

# Save linear relationship plots in hard drive
png(filename="MDG_and_MDA_linear_regression.png", units="px", width=7000, height=5000, res=600)
plot(dat2, main="Relationship between MDA and MDG (Scaled)", xlab="MDA", ylab="MDG", col="firebrick",
     las = 1, cex = 1.5, bty = "l", pch=16)
abline(lm(MDG~MDA, data=dat2), col="black")

dev.off()

# Linear regression calculation between MDA a MDG
lin <- lm(MDG~MDA, data=dat2)
lin

# Export linear regression coefficients in .txt format
z <- capture.output(lin)
lm <- cat(z, file="Linear_Model_Coefficients.txt", sep="\n", append=FALSE)

# Jaccard Index Calculation between MDA a DMG
a <- rownames(f1)
b <- rownames(f2)

jacard <- function(a, b){
  intersection = length(intersect(a, b))
  union = length(a) + length(b)
  return = (intersection/union)
}

ji <- jacard(a, b)
ji

# Export Jaccard Index results into hdd
jiz <- capture.output(ji)
jie <- cat(jiz, file="Jaccard_Index.txt", sep="\n", append=FALSE)

# Numbers of raster bands related to MDA
com <- f1
com

# Export of ordered MDA Importances
com_e <- capture.output(com)
com_u <- cat(com_e, file="MDA_Ordered.txt", sep="\n", append=FALSE)

# Numbers of raster bands related to MDG
com2 <- f2
com2

# Export of oredered MDG importances
com_e2 <- capture.output(com2)
com_u2 <- cat(com_e2, file="MDG_Ordered.txt", sep="\n", append=FALSE)
################################################################################################################################
################################################################################################################################

# Required Time for calculations
konec <- Sys.time()
cas <- konec - start
cas

# Save requiered time in .txt format
z_cas <- capture.output(cas)
u_cas <- cat(z_cas, file="Time_Requiered_Isolation_Forest_H2O.txt", sep="\n", append=FALSE)

##############################################################################################################################
##############################################################################################################################

