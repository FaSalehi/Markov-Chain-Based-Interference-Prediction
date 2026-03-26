# Markov Chain-Based Interference Prediction

This repository contains the source code of the paper "Reliable Interference Prediction and Management with Time-Correlated Traffic for URLLC", by Fateme Salehi, Aamir Mahmood, Nurul Huda Mahmood, and Mikael Gidlund, presented at IEEE GlobeCom 2023. [doi: 10.1109/GLOBECOM54140.2023.10437025](https://ieeexplore.ieee.org/abstract/document/10437025)

This paper focuses on interference prediction-based resource allocation, wherein our solution deviates from conventional average-based interference estimation schemes to a tail-behaviour framework. The interference space is modeled using a discrete-time Markov chain. We exploit the second-order Markov chain to capture the time
correlation among the samples. After estimating the statistics of the conditional interference distribution given current and previous observations, quantile prediction is applied to predict the next interference value, by considering the confidence level parameter to ensure QoS requirements. Eventually, the predicted interference is mitigated through efficient resource allocation. 

# Referencing
If you use this code in any way for research that results in publications, please cite our original article listed above.
