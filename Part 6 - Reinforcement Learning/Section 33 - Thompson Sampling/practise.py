# importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# importing dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')


# Thomson sampling model
import math
import random
N = 10000
d = 10
ad_selected = []
number_of_reward_0 = [0]*d
number_of_reward_1 = [0]*d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_random = 0
    for j in range(0, d):
        random_beta = random.betavariate(number_of_reward_1[j]+1, number_of_reward_0[j]+1)
        if random_beta > max_random:
            max_random = random_beta
            ad = j
    ad_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        number_of_reward_1[ad] += 1
    else:
        number_of_reward_0[ad] += 1
    total_reward += reward

print(total_reward)


# visualization
plt.hist(ad_selected)
plt.xlabel('Ads')
plt.ylabel('Number of times ad selected')
plt.show()