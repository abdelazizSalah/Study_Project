
## Task2 questions
- Should we use only the LabeledDataset, because it is the one with the pcap files => yes use only the pcap files
- Should we merge all the pcap files in one dataframe, or can we handle each file separatly. because it is not possible to load the whole data in one Dataframe, so should we use chunks ?
  - Depends on our implementation, but we should ask for the server access.
- It was mentioned in the task sheet that all flows should have the same length, but also in the next sentece it was mentioned that we should not lose information for the longer flows, so how is that possible ? should we truncate the flows to be of specific size or the data is generate in form of having same length for all flows.
  - Yes, all of them must have the same length, and also we should not truncate the flows. 
- If I have X and Y communicating on s7comm and modbus/TCP for example, a traffic flow would be from X to Y on S7comm, and another would be from Y to X on S7comm, or they both considered as one single traffic flow?
  - I assume that it will be bidirectional, so both X->Y and Y->X will be combined in single flow.
  - Yes they are bi-direction
- Is that what you want to see as traffic flow but using python script instead of using 
  - ![alt text](image.png) 
- Regarding the point of  **You should differentiate between flows containing
 packets under attack and those without attacks**, you mean that folder of attacks, and the folder of control, or you mean the attacker.pcap in each folder should be gathered together? because in the control we still have attacker.pcap also.
  - I assumed that you meant the folder of attacks should be handled separatly from the folder of control.
  - yes, we should handle each flow in the same window, and compare them on the same dataset. 
- Some packets we have undetected protocols for them, should we drop them? 
  - 
- What do you mean by direction?
  - I showed the src_ip and the dst_ip.
- How do I know if my results are correct or not?
  - Usually the logic should makes sense
  - But make sure that the code works. 
- Should we also include the direction in terms of MAC addresses, or only on level of IP. 
  - yes
- Should we try to understand the differnce between the used protocols in each dataset?
  - Focus on implementing the tasks, if you have time then we can do it later. 
- In spectra, the timestamp is strings, and when I looked into it, I found that it is in this form [0, 330, 654, ..., 200434218], and in the papers I read, it was not mentioned whether these are microseconds, but it was mentioned that the traffic is captured for more than 12 hours, 
  - All time stamps should be in the same format. 
  - And it is up to us to select which format.

### Task2.b
- By  "your piece of code should compute
 the Chebyshev distance between every two generated flows per time interval", do you mean for window size 2min for example, we should compute the Chebyshev distance for all flows? or should we compute the Chebyshev distance between 2mins from electra and 2mins from QUT?
  - Traffic flow within the same dataset only, and also within the same window size only
- Also we have selected three features "packet size", "direction", and "inter-arrival time", and Chebyshev is computing the maximum of the features difference, however, each of these features is completly different, so what makes sense for me is to compute the Chebyshev on the "packet size", so this is what you want? or should we perform normalization for example to make the features similar - but in this case we may drop the direction I think, because it is Nominal level, correct?
  - Select either "inter-arrival time" or "direction and packet size", and find a way to make direction and packet-size compitable with each other. 

- the submission, should be in python scripts, and we can use python libraries as long as the are not the main tasks.
# Answers:
- The header is everything before S7comm, and the payload of S7comm is also including S7comm header.
- The main goal is to reverse engineer  S7comm protocol
- We should know what are the attacked traffic, and what are the normal traffic to be able to diffrentiate between them, 