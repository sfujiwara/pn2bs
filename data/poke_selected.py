# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
# from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer

df = pd.read_csv('pokemon.csv')
col = [
    'name_jp',
    'hp',
    'attack',
    'block',
    'contact',
    'defense',
    'speed',
    'type1',
    'type2'
]

df[col].to_csv('poke_selected.csv', index=False)
v = DictVectorizer(sparse=False)
d = df[['type1', 'type2']].to_dict('record')
x = v.fit_transform(d)
