- In the first task, should we also use electra_s7comm ? because it was mentioned that we should handle only pcap files, however electra is .csv

## Task2 questions
- Should we use only the LabeledDataset, because it is the one with the pcap files.
- Should we merge all the pcap files in one dataframe, or can we handle each file separatly. 
- It was mentioned in the task sheet that all flows should have the same length, but also in the next sentece it was mentioned that we should not lose information for the longer flows, so how is that possible ? should we truncate the flows to be of specific size or the data is generate in form of having same length for all flows. 
- It is not possible to load the whole data in one Dataframe, so should we use chunks ?
- Is that what you want to see as traffic flow but using python script instead of using 
  - ![alt text](image.png) 