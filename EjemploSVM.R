#install.packages("caret")
#install.packages("dplyr")
#install.packages("doMC")
#install.packages("Boruta")
#install.packages("kernlab")

library(caret)
library(doMC)
library(dplyr)
library(Boruta)

set.seed(7)
inputData <- read.csv('sample_data/datav3.csv', sep = ',') 
head(inputData)

glimpse(inputData)

# Tabla de frecuencias 
table(inputData$result)

# Número de observaciones del set de datos
nrow(inputData)

# Detección si hay alguna fila incompleta
any(!complete.cases(inputData))

inputData$result = as.factor(inputData$result)

#Revisando la clase
table(inputData$result)

## Distribución de variables de clasificación
ggplot(data = inputData, aes(x = result, y = ..count.., fill = result)) +
  geom_bar() +
  scale_fill_manual(values = c("gray40", "orangered2")) +
  labs(title = "Clasificados con CVD") +
  theme_bw() +
  theme(legend.position = "bottom")

# Decidir si una variable es importante o no utilizando Boruta
boruta_output <- Boruta(result ~ ., data=na.omit(inputData), doTrace=2, maxRuns=500)  # perform Boruta search
names(boruta_output)

# Obtener variables significativas, incluyendo las tentativas
boruta_signif <- getSelectedAttributes(boruta_output, withTentative = TRUE)
print(boruta_signif)  

# Hacer un arreglo tentativo
roughFixMod <- TentativeRoughFix(boruta_output)
print(roughFixMod)
attStats(boruta_output)
boruta_signif <- getSelectedAttributes(roughFixMod)
print(boruta_signif)

# Puntuación de la importancia de las variables
imps <- attStats(roughFixMod)
imps2 = imps[imps$decision != 'Rejected', c('meanImp', 'decision')]
head(imps2[order(-imps2$meanImp), ])  # descending sort

set.seed(123)

# Se crean los índices de las observaciones de entrenamiento
train <- createDataPartition(y = inputData$result, p = 0.7, list = FALSE, times = 1)
datos_train <- inputData[train, ]
datos_test  <- inputData[-train, ]

# Resumen del set de datos train
glimpse(datos_train)

# Resumen del set de datos train
glimpse(datos_test)

library(doMC)
registerDoMC(cores = 4)

particiones  <- 10
repeticiones <- 5

# Hiperparámetros
hiperparametros <- expand.grid(sigma = c(0.001, 0.01, 0.1, 0.5, 1),
                               
set.seed(123)
seeds <- vector(mode = "list", length = (particiones * repeticiones) + 1)
for (i in 1:(particiones * repeticiones)) {
  seeds[[i]] <- sample.int(1000, nrow(hiperparametros))
}
seeds[[(particiones * repeticiones) + 1]] <- sample.int(1000, 1)

control_train <- trainControl(method = "repeatedcv", number = particiones,
                              repeats = repeticiones, seeds = seeds,
                              returnResamp = "all", verboseIter = FALSE,
                              classProbs = TRUE, allowParallel = TRUE)

set.seed(342)


# # Entrenamiento del SVM con un kernel radial y optimización del hiperparámetro C y sigma
svmT <- train(result ~ edad + calcio_en_sangre + taquicardia_sinusal_ecg,
              data = datos_train,
              method = "svmRadial",
              tuneGrid = hiperparametros,
              metric = "Accuracy",
              trControl = control_train)

# Resultado del entrenamiento
svmT

# # Entrenamiento del SVM con un kernel lineal y optimización del hiperparámetro C y sigma
svmT <- train(result ~ edad + calcio_en_sangre + taquicardia_sinusal_ecg,
              data = datos_train,
              method = "svmRadial",
              tuneGrid = hiperparametros,
              metric = "Accuracy",
              preProc = c("center", "scale"), #estandarizacion de los datos
              trControl = control_train)

# Resultado del entrenamiento
svmT

# Valores de validación (Accuracy y Kappa) obtenidos en cada partición y repetición.
svmT$resample %>% head(10)
summary(svmT$resample$Accuracy)

# Evolución del accuracy en funcion del valor de coste en validacion cruzada
plot(svmT)

ggplot(data = svmT$resample,
       aes(x = as.factor(C), y = Accuracy, color = as.factor(C))) +
  geom_boxplot(outlier.shape = NA, alpha = 0.6) +
  geom_jitter(width = 0.2, alpha = 0.6) +
  # Línea horizontal en el accuracy basal
  geom_hline(yintercept = 0.8, linetype = "dashed") +
  labs(x = "C") +
  theme_bw() + theme(legend.position = "none")


ggplot(svmT, highlight = TRUE) +
  labs(title = "Evolución del accuracy del modelo en función de C") +
  theme_bw()

#Predicciones
p <- predict(svmT, datos_test)
predicciones_prob <- predict(svmT, newdata = datos_test, type = "prob")
predicciones_prob
predicciones_raw <- predict(svmT, newdata = datos_test, type = "raw")
predicciones_raw

#EVALUACION DEL MODELO
###confusion Matrix
confusionMatrix(predicciones_raw, datos_test$result)

# Calcular la exactitud del modelo
porcentaje <- mean(p == datos_test$result)*100
paste("Porcentaje de predicción SVM:",round(porcentaje,2))

# Error de test
error_test <- mean(p != datos_test$result)
paste("El error de test del modelo SVM:", round(error_test*100, 2), "%")

#Salvando el modelo
saveRDS(svmT,"svmT.rds")