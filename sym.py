from osim.env import ProstheticsEnv
import random
import numpy as np
import pandas as pd

env = ProstheticsEnv(visualize=True)

alpha = 0.01
gamma = 0.9
delta = 0


def fu(g):
    for i in range(19):
        if s[i] == g:
            if a[i] == 0:
                s[i] = g + random.randint(20, 40)
                a[i] = 1
                f[i] = random.uniform(0.3, 1)
            else:
                s[i] = g + random.randint(5, 30)
                a[i] = 0
                f[i] = 0
    return f


def value(state, w):
    return np.dot(state, w.transpose())

w = np.array([0.]*177)
df = pd.DataFrame()

for i in range(177):
    st = "val_" + str(i)
    df[st] = pd.Series([])
df['target'] = pd.Series([])

count = 0

for j in range(1000):
    s = [0] * 19
    a = [0] * 19
    f = [0] * 19
    for i in range(19):
        a[i] = random.randint(0, 1)
    trace = []
    reward = 0
    le = 0
    observation = env.reset()
    for i in range(300):
        le += 1
        a = fu(i)
        x = observation + a
        observation, reward, done, info = env.step(a, project=True)
        x = (x, reward)
        trace.append(x)
        if done:
            break

    print("length: ", le)

    g = 0
    for i in range(len(trace)-1, -1, -1):
        g += trace[i][1]
        state = np.array(trace[i][0]+[g])
        df.loc[count] = state
        count += 1

    df.to_csv('cos.csv')


