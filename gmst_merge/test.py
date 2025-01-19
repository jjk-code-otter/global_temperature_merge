import random

from combinator import split_list, make_random_tree
import matplotlib.pyplot as plt
import numpy as np



lst = ['a', 'b' ,'c','d','e','f','g','h','i','j','k']


print(split_list(lst, 4))


for i in range(100):
    print(make_random_tree(lst))