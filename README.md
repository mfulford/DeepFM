# DeepFM

#### DeepFM<sup>[1]</sup> implemented as a custom keras layer


## Getting started: 

Build a `DeepFM` model in a few lines of code:

```python
from deepFM_keras import LayerDeepFM
from keras.layers import Input

inputTensor = Input(shape=(len_input,))

deepFM_out = LayerDeepFM(field_lengths, one_hot=one_hot, embedding_size=10, deepHidenLayers=[50, 50],
                          deep=True, fm=True, return_embedding=False, input_shape=(len_input,))(inputTensor)
                      
out = Dense(1, activation="sigmoid", trainable=True)(deepFM_out)
model = Model(inputTensor, out)
```

`deepFM_keras` accepts the input either one hot encoded or integer encoded. Input dimensions are required to compile model.
When the input is one hot encoded:
```python
one_hot = True
len_input = sum(field_lengths.values())
```
or when the input is integer encoded:
```python
one_hot = False
len_input = len(field_lengths)
``` 

### Output options:

DeepFM involves a factorisation machine component (FM) and deep neural network component. In addition to returning the DeepFM output, `deepFM_keras` can be used to return simply the FM output or deep output. Like the previous example, set `return_embedding = False` to return the final layer output and specify whether to include FM and or deep components with  `deep` and `fm` parameters. NB if `fm = True` and `deep = True`, the output is the DeepFM output in the first example. 

```python
# Return FM output when fm = True and deep = False:
FM_out = LayerDeepFM(field_lengths, one_hot=one_hot, embedding_size=10, deepHidenLayers=[50, 50],
                     deep=False, fm=True, return_embedding=False, input_shape=(len_input,))(inputTensor)

# Return deep output when fm = False and deep = True:
deep_out = LayerDeepFM(field_lengths, one_hot=one_hot, embedding_size=10, deepHidenLayers=[50, 50],
                       deep=True, fm=False, return_embedding=False, input_shape=(len_input,))(inputTensor)
``` 


`deepFM_keras` can also be used to generate deep embeddings, FM embeddings or concatenated deep + FM embeddings to use in other models. In this case set `return_embedding = True` and specify whether to include FM and/or deep components with  `deep` and `fm` parameters.

```python
# Return FM embedding when fm = True and deep = False:
FM_embed = LayerDeepFM(field_lengths, one_hot=one_hot, embedding_size=10, deepHidenLayers=[50, 50],
                       deep=False, fm=True, return_embedding=True, input_shape=(len_input,))(inputTensor)

# Return deep embedding when fm = False and deep = True:
deep_embed = LayerDeepFM(field_lengths, one_hot=one_hot, embedding_size=10, deepHidenLayers=[50, 50],
                         deep=True, fm=False, return_embedding=True, input_shape=(len_input,))(inputTensor)
                         
# Return concatenated deep and FM embeddings when fm = True and deep = True:
deepFM_embed = LayerDeepFM(field_lengths, one_hot=one_hot, embedding_size=10, deepHidenLayers=[50, 50],
                            deep=True, fm=True, return_embedding=True, input_shape=(len_input,))(inputTensor)
``` 

Any of the outputs from `deepFM_keras` (`deepFM_embed`, `deep_embed`, `FM_embed`, `deep_out`, `FM_out`, `deepFM_out`) can be used as input for an output layer.
e.g

```python
out = Dense(1, activation="sigmoid", trainable=True)(FM_out)
``` 

Equally, the algorithm works with multiclass classification

```python
num_classes = 3
out = Dense(num_classes, activation="softmax", trainable=True)(FM_out) 
``` 
as well as regression

```python
out = Dense(1, activation="linear", trainable=True)(FM_out)
``` 

Finally the model is compiled with the appropriate loss:

```python
model = Model(inputTensor, out) 
model.compile(loss=loss, optimizer=opt,metrics=metrics)
``` 

### Input Examples:

5 fields with dimensions 3,3,2,2,2 respectively. 

```python
field_lengths = {0:3, 1:3, 2:2, 3:2, 4:2}
```

One hot encoded example with 3 rows:
```python
one_hot = True
X = np.array([
        [0,0,1, 0,0,1, 0,1, 0,1, 1,0],
        [1,0,0, 0,0,0, 1,0, 1,0, 1,0],
        [0,1,0, 1,0,0, 0,1, 0,0, 0,1]
    ])
```
or equivalent integer encoded:
```python
one_hot = False
X = np.array([
        [3, 3, 2, 2, 1],
        [1, 0, 1, 1, 1],
        [2, 1, 2, 0, 2]
    ])
```


# Reference
[1] *DeepFM: A Factorization-Machine based Neural Network for CTR Prediction*, Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He.
