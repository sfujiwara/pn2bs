# -*- coding: utf-8 -*-

import pandas as pd


df = pd.read_csv('data/pokemon.csv')
ab1 = []
ab2 = []
abh = []

for i, row in df.iterrows():
    tmp = row.ability1.split('\n')
    if len(tmp) == 3:
        print tmp[0], tmp[1], tmp[2]
        ab1.append(tmp[0])
        ab2.append(tmp[1])
        h = tmp[2].strip().replace('(', '').replace(')', '')
        abh.append(h)
    elif len(tmp) == 2:
        ab1.append(tmp[0])
        if tmp[1][-1] == ')':
            ab2.append(None)
            h = tmp[1].strip().replace('(', '').replace(')', '')
            abh.append(h)
        else:
            ab2.append(tmp[1])
            abh.append(None)
    elif len(tmp) == 1:
        ab1.append(tmp[0])
        ab2.append(None)
        abh.append(None)
    else:
        'ERROR!!!!!!!!!!!!!!!!'
        break

df['ability1'] = ab1
df['ability2'] = ab2
df['hidden_ability'] = abh
df.to_csv('pokemon2.csv')