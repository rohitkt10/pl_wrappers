# PyTorch Lightning wrappers 

The goal of this project is to write `pytorch_lightning` based wrappers around regular `torch.nn.Module` that follow the approach of initializing, training and testing models defined in `keras`, i.e., we would like to be able to define a model with optimizers, metrics and loss functions, train the model with a `fit` function and test the model with an `evaluate` method. 

The target models for this work are standard supervised DNNs and `gpytorch` based GP models. Extensions to unsupervised models such as autoencoders will be considered later. 

Example: 

```python
model = Model()  ## subclass of PLkerasModel
model.compile(optimizer, baselr, lossfn, metrics)  ## compile the model with an optimizer, loss function and a dictionary of metrics
model.fit(x, y, validation_split=0.2, batch_size, epochs)  ## fit the model by passing data as either `torch.Tensor` or `numpy.ndarray`. 
test_results = model.evaluate(x_test, y_test) 
```
