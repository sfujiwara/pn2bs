# pokemon-name to base-stats

[![Heroku Deploy](https://www.herokucdn.com/deploy/button.png)](https://heroku.com/deploy)

```sh
heroku create --buildpack https://github.com/kennethreitz/conda-buildpack.git pn2bs
git push heroku master
```

## Training and Prediction

### Training

```sh
python pn2bs.py
```

### Prediction

```python
import predictor

res = predictor.predict('テンソルフロー')
print res
```

### Run TensorBoard

```sh
tensorboard --logdir=<absolute path>/pokename2basestats/log
```

## Use as Web API

### Run API Server

Deploy to Heroku or

```sh
python main.py
```

### Request

Send GET request as

```
https://pn2bs.herokuapp.com/?name=テンソルフロー
```

### Response

```json
{
  "name": "テンソルフロー",
  "hp": 100,
  "attack": 100,
  "block": 100,
  "contact": 100,
  "defense": 100,
  "speed": 100,
  "type1": "フェアリー",
  "type2": "あく"
}
```
