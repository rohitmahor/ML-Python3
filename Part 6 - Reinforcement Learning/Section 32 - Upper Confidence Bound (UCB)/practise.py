# importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# importing dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')


# random selection
import random
# N = 10000
# d = 10
# ad_selected = []
# total_reward = 0
# for n in range(0, N):
#     ad = random.randrange(d)
#     ad_selected.append(ad)
#     reward = dataset.values[n, ad]
#     total_reward += reward

# UCB model
import math
N = 10000
d = 10
ad_selected = []
number_of_selection = [0]*d
sums_of_reward = [0]*d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_ucb = 0
    for j in range(0, d):
        if number_of_selection[j] > 0:
            average_reward = sums_of_reward[j]/number_of_selection[j]
            delta_i = math.sqrt(3 / 2 * math.log(n + 1) / number_of_selection[j])
            ucb = average_reward + delta_i
        else:
            ucb = 1e400
        if ucb > max_ucb:
            max_ucb = ucb
            ad = j
    ad_selected.append(ad)
    number_of_selection[ad] += 1
    reward = dataset.values[n, ad]
    total_reward += reward
    sums_of_reward[ad] += reward

print(total_reward)
# visualization
plt.hist(ad_selected)
plt.xlabel('Ads')
plt.ylabel('Number of times ad selected')
plt.show()