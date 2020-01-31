#R version 3.6.1

library(NbClust)

packageVersion("NbClust")
# 3.0

iris <- read.csv("iris.csv")
res <- NbClust(iris[,-5], method="complete", index="all")
# 3
iris.scaled <- scale(iris[,-5])
res <- NbClust(iris.scaled, method="complete", index="all")
# 3

wine <- read.csv("wine.csv")
res <- NbClust(wine[,-14], method="complete", index="all")
# 2
wine.scaled <- scale(wine[,-14])
res <- NbClust(wine.scaled, method="complete", index="all")
# 3

cancer <- read.csv("cancer.csv")
res <- NbClust(cancer[,-31], method="complete", index="all")
# 4
cancer.scaled <- scale(cancer[,-31])
res <- NbClust(cancer.scaled, method="complete", index="all")
# 4