
library(caret)
library(dplyr)
library(randomForest)
library(xgboost)
library(plumber)

#* @apiTitle API clasificación de una enfermedad Cardiovascular y predicción de Arritmia.
#* @apiDescription API clasificación de una CVD y predicción de Arritmia a través de RandomForest, Support Vector Machine y XGBoost. 



#' Log system time, request method and HTTP user agent of the incoming request
#' @filter logger
function(req){
  cat("System time:", as.character(Sys.time()), "\n",
      "Request method:", req$REQUEST_METHOD, req$PATH_INFO, "\n",
      "HTTP user agent:", req$HTTP_USER_AGENT, "@", req$REMOTE_ADDR, "\n")
  plumber::forward()
}

#' @filter cors
cors <- function(req, res) {
  
  res$setHeader("Access-Control-Allow-Origin", "*")
  
  if (req$REQUEST_METHOD == "OPTIONS") {
    res$setHeader("Access-Control-Allow-Methods","*")
    res$setHeader("Access-Control-Allow-Headers", req$HTTP_ACCESS_CONTROL_REQUEST_HEADERS)
    res$status <- 200 
    return(list())
  } else {
    plumber::forward()
  }
  
}


#####################################################################################
## API CLASIFICACION ENFERMEDAD CARDIOVASCULAR
#####################################################################################
# Core function RANDOM FOREST
# define parameters with type and description
# name endpoint
# return output as html/text
# specify 200 (okay) return

#* @param edad Edad del paciente
#* @param creatinine Creatinina del paciente.
#* @param old_peak Old peak electrocardiograma.
#* @param rest_ecf Resultado electrocardiograma en reposo
#* @get /clasificaRF
#* @post /clasificaRF

clasificaRF <- function(edad, creatinine, old_peak, rest_ecf) {
  #Load Model
  modelRF <- readRDS("rf4.rds")
  # make data frame from numeric parameters
  input_data_num <<- data.frame(creatinine, old_peak, stringsAsFactors = FALSE)
  # and make sure they really are numeric
  input_data_num <<- as.data.frame(t(sapply(input_data_num, as.numeric)))
  
  input_data_int <<- data.frame(edad, rest_ecf, stringsAsFactors = FALSE)
  # and make sure they really are numeric
  input_data_int <<- as.data.frame(t(sapply(input_data_int, as.integer)))
  # combine into one data frame
  features <<- as.data.frame(cbind(input_data_num, input_data_int))
  
  # validation for parameter
  if (any(is.na(features))) {
    res$status <- 400
    res$body <- "Parameters have to be numeric or integers"
  }
  
  # predict and return result
  pred_cvd1 <<- predict(modelRF, features, type="response")
  pred_cvd2 <<- predict(modelRF, features,type = "prob")
  
  if(pred_cvd1=='no')
  {
    paste("RF - Clasificado CVD = No - Predicción = ", as.character(round(pred_cvd2[,1]*100,2)))
  }
  else {
    paste("RF - Clasificado CVD = Si - Predicción = ", as.character(round(pred_cvd2[,2]*100,2)))
  }
} 



#############################################
### API SUPPORT VECTOR MACHINE
############################################
# Core function RANDOM FOREST
# define parameters with type and description
# name endpoint
# return output as html/text
# specify 200 (okay) return

#* @param edad Edad del paciente
#* @param creatinine Creatinina del paciente.
#* @param old_peak Old peak electrocardiograma.
#* @param rest_ecf Resultado electrocardiograma en reposo
#* @get /clasificaSVM
#* @post /clasificaSVM

clasificaSVM <- function(edad, creatinine, old_peak, rest_ecf) {
  #Load Model
  modelSVM <- readRDS("svm4.rds")
  # make data frame from numeric parameters
  input_data_num <<- data.frame(creatinine, old_peak, stringsAsFactors = FALSE)
  # and make sure they really are numeric
  input_data_num <<- as.data.frame(t(sapply(input_data_num, as.numeric)))
  
  input_data_int <<- data.frame(edad, rest_ecf, stringsAsFactors = FALSE)
  # and make sure they really are numeric
  input_data_int <<- as.data.frame(t(sapply(input_data_int, as.integer)))
  # combine into one data frame
  features <<- as.data.frame(cbind(input_data_num, input_data_int))
  
  # validation for parameter
  if (any(is.na(features))) {
    res$status <- 400
    res$body <- "Parameters have to be numeric or integers"
  }
  
  # predict and return result
  pred_cvd1 <<- predict(modelSVM, features, type="raw")
  pred_cvd2 <<- predict(modelSVM, features,type = "prob")
  if(pred_cvd1=='no')
  {
    paste("SVM - Clasificado CVD = No - Predicción = ", as.character(round(pred_cvd2[,1]*100,2)))
  }
  else {
    paste("SVM - Clasificado CVD = Si - Predicción = ", as.character(round(pred_cvd2[,2]*100,2)))
  }
} 


#############################################
### API XGBOOST
############################################
# Core function xgboOst
# define parameters with type and description
# name endpoint
# return output as html/text
# specify 200 (okay) return

#* @param edad Edad del paciente
#* @param creatinine Creatinina del paciente.
#* @param old_peak Old peak electrocardiograma.
#* @param rest_ecf Resultado electrocardiograma en reposo
#* @post /clasificaXGBF
#* @get /clasificaXGBF

clasificaXGBF <- function(edad, creatinine, old_peak, rest_ecf) {
  #Load Model
  modelXBGF <- readRDS("xgb4.rds")
  # make data frame from numeric parameters
  input_data_num <<- data.frame(creatinine, old_peak, stringsAsFactors = FALSE)
  # and make sure they really are numeric
  input_data_num <<- as.data.frame(t(sapply(input_data_num, as.numeric)))
  
  input_data_int <<- data.frame(edad, rest_ecf, stringsAsFactors = FALSE)
  # and make sure they really are numeric
  input_data_int <<- as.data.frame(t(sapply(input_data_int, as.integer)))
  # combine into one data frame
  features <<- as.data.frame(cbind(input_data_num, input_data_int))
  
  # validation for parameter
  if (any(is.na(features))) {
    res$status <- 400
    res$body <- "Parameters have to be numeric or integers"
  }
  
  # predict and return result
  pred_cvd1 <<- predict(modelXBGF, features, type="raw")
  pred_cvd2 <<- predict(modelXBGF, features,type = "prob")
  if(pred_cvd1=='no')
  {
    paste("XGBF - Clasificado CVD = No - Predicción = ", as.character(round(pred_cvd2[,1]*100,2)))
  }
  else {
    paste("XGBF - Clasificado CVD = Si - Predicción = ", as.character(round(pred_cvd2[,2]*100,2)))
  }
} 


###############################################################################################################
## APIS ARRITMIA

#############################################
### API RF ARRITMIA RANDOM FOREST
############################################

#* @param edad Edad del paciente
#* @param ejection_fraction Ejection fracion electrocardiograma.
#* @param taquicardia_sinusal_ecg Taquicardia sinusal electrocardiograma
#* @post /arritmiaRF
#* @get /arritmiaRF

arritmiaRF <- function(edad, ejection_fraction, taquicardia_sinusal_ecg) {
  #Load Model
  modelA_RF <- readRDS("rfA.rds")
  input_data_int <<- data.frame(edad, ejection_fraction, taquicardia_sinusal_ecg, stringsAsFactors = FALSE)
  # and make sure they really are numeric
  input_data_int <<- as.data.frame(t(sapply(input_data_int, as.integer)))
  # combine into one data frame
  features <<- as.data.frame(cbind(input_data_int))
  
  # validation for parameter
  if (any(is.na(features))) {
    res$status <- 400
    res$body <- "Parameters have to be numeric or integers"
  }
  
  # predict and return result
  pred_cvd1 <<- predict(modelA_RF, features, type="response")
  pred_cvd2 <<- predict(modelA_RF, features,type = "prob")
  if(pred_cvd1=='no')
  {
    list(
       No = paste(round(pred_cvd2[,1]*100,2) ),
       Si = paste(round(pred_cvd2[,2]*100,2) ) 
    )
    
  }
  else {
    list(
      Si = paste(round(pred_cvd2[,2]*100,2) ),
      No = paste(round(pred_cvd2[,1]*100,2) )  
    )
    }
    
} 

#############################################
### API RF ARRITMIA SVM
############################################

#* @param edad Edad del paciente
#* @param ejection_fraction Ejection fracion electrocardiograma.
#* @param taquicardia_sinusal_ecg Taquicardia sinusal electrocardiograma
#* @post /arritmiaSVM
#* @get /arritmiaSVM

arritmiaSVM <- function(edad, ejection_fraction, taquicardia_sinusal_ecg) {
  #Load Model
  modelA_SVM <- readRDS("svmA.rds")
  input_data_int <<- data.frame(edad, ejection_fraction, taquicardia_sinusal_ecg, stringsAsFactors = FALSE)
  # and make sure they really are numeric
  input_data_int <<- as.data.frame(t(sapply(input_data_int, as.integer)))
  # combine into one data frame
  features <<- as.data.frame(cbind(input_data_int))
  
  # validation for parameter
  if (any(is.na(features))) {
    res$status <- 400
    res$body <- "Parameters have to be numeric or integers"
  }
  
  # predict and return result
  pred_cvd1 <<- predict(modelA_SVM, features, type="raw")
  pred_cvd2 <<- predict(modelA_SVM, features,type = "prob")
  if(pred_cvd1=='no')
  {
    list(
      No = paste(round(pred_cvd2[,1]*100,2) ),
      Si = paste(round(pred_cvd2[,2]*100,2) ) 
    )
    
  }
  else {
    list(
      Si = paste(round(pred_cvd2[,2]*100,2) ),
      No = paste(round(pred_cvd2[,1]*100,2) )  
    )
  }
  
} 

#############################################
### API  ARRITMIA XGBF
############################################

#* @param edad Edad del paciente
#* @param ejection_fraction Ejection fracion electrocardiograma.
#* @param taquicardia_sinusal_ecg Taquicardia sinusal electrocardiograma
#* @post /arritmiaXGBF
#* @get /arritmiaXGBF

arritmiaXGBF <- function(edad, ejection_fraction, taquicardia_sinusal_ecg) {
  #Load Model
  modelA_XGBF <- readRDS("xgbA.rds")
  input_data_int <<- data.frame(edad, ejection_fraction, taquicardia_sinusal_ecg, stringsAsFactors = FALSE)
  # and make sure they really are numeric
  input_data_int <<- as.data.frame(t(sapply(input_data_int, as.integer)))
  # combine into one data frame
  features <<- as.data.frame(cbind(input_data_int))
  
  # validation for parameter
  if (any(is.na(features))) {
    res$status <- 400
    res$body <- "Parameters have to be numeric or integers"
  }
  
  # predict and return result
  pred_cvd1 <<- predict(modelA_XGBF, features, type="raw")
  pred_cvd2 <<- predict(modelA_XGBF, features,type = "prob")
  if(pred_cvd1=='no')
  {
    list(
      No = paste0(round(pred_cvd2[,1]*100,2) ),
      Si = paste0(round(pred_cvd2[,2]*100,2) ) 
    )
    
  }
  else {
    list(
      Si = paste(round(pred_cvd2[,2]*100,2) ),
      No = paste(round(pred_cvd2[,1]*100,2) )  
    )
  }
  
} 

#* Example of customizing graphical output
#* @serializer png list(width = 400, height = 500)
#* @get /
#function(){
 # plot(1:10, xlab = "Etiqueta del eje X", ylab = "Etiqueta del eje Y", main="Hola mundo")
#}



###############################################################################################################
## APIS TAQUICARDIA

#############################################
### API RF ARRITMIA RANDOM FOREST
############################################

#* @param edad Edad del paciente
#* @param calcio_en_sangre Calcio en sangre.
#* @param taquicardia_sinusal_ecg Taquicardia sinusal electrocardiograma
#* @post /taquicardiaRF
#* @get /taquicardiaRF

taquicardiaRF <- function(edad, calcio_en_sangre, taquicardia_sinusal_ecg) {
  #Load Model
  modelT_RF <- readRDS("rfT.rds")
  input_data_int <<- data.frame(edad, calcio_en_sangre, taquicardia_sinusal_ecg, stringsAsFactors = FALSE)
  # and make sure they really are numeric
  input_data_int <<- as.data.frame(t(sapply(input_data_int, as.integer)))
  # combine into one data frame
  features <<- as.data.frame(cbind(input_data_int))
  
  # validation for parameter
  if (any(is.na(features))) {
    res$status <- 400
    res$body <- "Parameters have to be numeric or integers"
  }
  
  # predict and return result
  pred_cvd1 <<- predict(modelT_RF, features, type="response")
  pred_cvd2 <<- predict(modelT_RF, features,type = "prob")
  if(pred_cvd1=='no')
  {
    list(
      No = paste(round(pred_cvd2[,1]*100,2) ),
      Si = paste(round(pred_cvd2[,2]*100,2) ) 
    )
    
  }
  else {
    list(
      Si = paste(round(pred_cvd2[,2]*100,2) ),
      No = paste(round(pred_cvd2[,1]*100,2) )  
    )
  }
  
} 


#############################################
### API SVM TAQUICARDIA
############################################

#* @param edad Edad del paciente
#* @param calcio_en_sangre Calcio en sangre.
#* @param taquicardia_sinusal_ecg Taquicardia sinusal electrocardiograma
#* @post /taquicardiaSVM
#* @get /taquicardiaSVM

taquicardiaSVM <- function(edad, calcio_en_sangre, taquicardia_sinusal_ecg) {
  #Load Model
  modelT_SVM <- readRDS("svmT.rds")
  input_data_int <<- data.frame(edad, calcio_en_sangre, taquicardia_sinusal_ecg, stringsAsFactors = FALSE)
  # and make sure they really are numeric
  input_data_int <<- as.data.frame(t(sapply(input_data_int, as.integer)))
  # combine into one data frame
  features <<- as.data.frame(cbind(input_data_int))
  
  # validation for parameter
  if (any(is.na(features))) {
    res$status <- 400
    res$body <- "Parameters have to be numeric or integers"
  }
  
  # predict and return result
  pred_cvd1 <<- predict(modelT_SVM, features, type="raw")
  pred_cvd2 <<- predict(modelT_SVM, features,type = "prob")
  if(pred_cvd1=='no')
  {
    list(
      No = paste(round(pred_cvd2[,1]*100,2) ),
      Si = paste(round(pred_cvd2[,2]*100,2) ) 
    )
    
  }
  else {
    list(
      Si = paste(round(pred_cvd2[,2]*100,2) ),
      No = paste(round(pred_cvd2[,1]*100,2) )  
    )
  }
  
} 


#############################################
### API XGBF TAQUICARDIA
############################################

#* @param edad Edad del paciente
#* @param calcio_en_sangre Calcio en sangre.
#* @param taquicardia_sinusal_ecg Taquicardia sinusal electrocardiograma
#* @post /taquicardiaXGBF
#* @get /taquicardiaXGBF

taquicardiaXBGF <- function(edad, calcio_en_sangre, taquicardia_sinusal_ecg) {
  #Load Model
  modelT_XGBF <- readRDS("xgbT.rds")
  input_data_int <<- data.frame(edad, calcio_en_sangre, taquicardia_sinusal_ecg, stringsAsFactors = FALSE)
  # and make sure they really are numeric
  input_data_int <<- as.data.frame(t(sapply(input_data_int, as.integer)))
  # combine into one data frame
  features <<- as.data.frame(cbind(input_data_int))
  
  # validation for parameter
  if (any(is.na(features))) {
    res$status <- 400
    res$body <- "Parameters have to be numeric or integers"
  }
  
  # predict and return result
  pred_cvd1 <<- predict(modelT_XGBF, features, type="raw")
  pred_cvd2 <<- predict(modelT_XGBF, features,type = "prob")
  if(pred_cvd1=='no')
  {
    list(
      No = paste(round(pred_cvd2[,1]*100,2) ),
      Si = paste(round(pred_cvd2[,2]*100,2) ) 
    )
    
  }
  else {
    list(
      Si = paste(round(pred_cvd2[,2]*100,2) ),
      No = paste(round(pred_cvd2[,1]*100,2) )  
    )
  }
  
} 
