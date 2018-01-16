library(igraph)

sNetworksDir = "networks/"

cSizes = c(100,1000,10000)
cNetworkModels = c("ER", "BA", "WS")
cBApowers = c(0.5, 1, 1.5)
cERprobs = c(0.05, 0.1, 0.25)
cWSprobs = c(0.01, 0.1, 0.25)

for(cNetworkModel in cNetworkModels) {
  for(cSize in cSizes) {
    if(cNetworkModel == "ER") {
      for(cERprob in cERprobs) {
        gGraph <- erdos.renyi.game(cSize, cERprob)
        write_graph(gGraph, file=paste(sNetworksDir, cNetworkModel, "-", cSize, "-", cERprob, ".csv", sep=""), format="edgelist")
        print(sprintf("Nodes: %s, edges: %s", vcount(gGraph), ecount(gGraph)))
      }
    } else if (cNetworkModel == "BA") {
      for(cBApower in cBApowers) {
        gGraph <- barabasi.game(cSize, cBApower, directed=F)
        write_graph(gGraph, file=paste(sNetworksDir, cNetworkModel, "-", cSize, "-", cBApower, ".csv", sep=""), format="edgelist")
        print(sprintf("Nodes: %s, edges: %s", vcount(gGraph), ecount(gGraph)))
      }
    } else if (cNetworkModel == "WS") {
      for(cWSprob in cWSprobs) {
        gGraph <- sample_smallworld(1, cSize, 2, cWSprob)
        write_graph(gGraph, file=paste(sNetworksDir, cNetworkModel, "-", cSize, "-", cWSprob, ".csv", sep=""), format="edgelist")
        print(sprintf("Nodes: %s, edges: %s", vcount(gGraph), ecount(gGraph)))
      }
    }
  }
}