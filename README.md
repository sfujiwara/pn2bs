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
tensorboard --logdir=<absolute path>/tflogs
```

## Use as Web API

### Run API Server

Deploy to Heroku or

```sh
python main.py
```

### Request

For example, send GET request as

```
https://pn2bs.herokuapp.com/pred?name=テンソルフロー
```

### Response

```json
{
  "name": "テンソルフロー",
  "hp": 92,
  "attack": 101,
  "block": 84,
  "contact": 85,
  "defense": 65,
  "speed": 72,
  "type1": "フェアリー",
  "type2": "あく"
 }
```
