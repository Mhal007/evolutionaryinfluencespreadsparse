library(igraph)

# directory for resulting networks
sNetworksDir = "networks/"

# number of nodes
cSizes = c(1000)

# which models to include
cNetworkModels = c("ER", "BA", "WS")

# models' parameters
cBApowers = c(1)
cERprobs = c(0.001)
cWSprobs = c(0.05)

# how many networks' instances to generate?
nNumOfAttempts <- 10

# shall I randomize weights too?
bWeights <- F

for(i in 1:nNumOfAttempts) {
  for(cNetworkModel in cNetworkModels) {
    for(cSize in cSizes) {
      if(cNetworkModel == "ER") {
        for(cERprob in cERprobs) {
          gGraph <- erdos.renyi.game(cSize, cERprob)
          if(bWeights == T) {
            E(gGraph)$weight <- round(runif(length(E(gGraph)), 0.01, 1), 4)
          }
          #plot(gGraph, vertex.label=NA, vertex.size=2)
          write_graph(gGraph, file=paste(sNetworksDir, cNetworkModel, "-", cSize, "-", cERprob, "-", i, ".csv", sep=""), format="ncol")
          print(sprintf("Nodes: %s, edges: %s", vcount(gGraph), ecount(gGraph)))
        }
      } else if (cNetworkModel == "BA") {
        for(cBApower in cBApowers) {
          gGraph <- barabasi.game(cSize, cBApower, directed=F)
          if(bWeights == T) {
            E(gGraph)$weight <- round(runif(length(E(gGraph)), 0.01, 1), 4)
          }
          #plot(gGraph, vertex.label=NA, vertex.size=2)
          write_graph(gGraph, file=paste(sNetworksDir, cNetworkModel, "-", cSize, "-", cBApower, "-", i, ".csv", sep=""), format="ncol")
          print(sprintf("Nodes: %s, edges: %s", vcount(gGraph), ecount(gGraph)))
        }
      } else if (cNetworkModel == "WS") {
        for(cWSprob in cWSprobs) {
          gGraph <- sample_smallworld(1, cSize, 2, cWSprob)
          if(bWeights == T) {
            E(gGraph)$weight <- round(runif(length(E(gGraph)), 0.01, 1), 4)
          }
          #plot(gGraph, vertex.label=NA, vertex.size=2)
          write_graph(gGraph, file=paste(sNetworksDir, cNetworkModel, "-", cSize, "-", cWSprob, "-", i, ".csv", sep=""), format="ncol")
          print(sprintf("Nodes: %s, edges: %s", vcount(gGraph), ecount(gGraph)))
        }
      }
    }
  }
}
