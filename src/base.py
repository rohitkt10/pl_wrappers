import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl

__all__ = ['PLKerasModel']

class PLKerasModel(pl.LightningModule):
    @property
    def mode(self):
        if self.training:
            return "train"
        else:
            return "eval"

    @property
    def device(self):
        return list(self.parameters())[0].device

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        self._dtype = dtype

    @property
    def metric_names(self):
        if len(self.metrics) > 0:
            return list(self.metrics.keys())
        else:
            return None

    def __init__(self, dtype=torch.float32):
        super().__init__()
        self._dtype = dtype
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def configure_optimizers(self):
        if hasattr(self, "lr_scheduler"):
            return [self.optimizer], [self.lr_scheduler]
        else:
            return self.optimizer
    
    def compile(self, optimizer, 
                lossfn, 
                baselr=1e-3, 
                metrics={}, 
                lr_scheduler_type=None, 
                lr_scheduler_options=None, 
                optimizer_kwargs={}):
        """
        Arguments:
        
            1. optimizer (Required) - The optimizer should a pointer to the `type` of torch optimizer you wish to use. For example,
                            if you wish to use the Adam optimizer, set optimizer=torch.optim.Adam. 
            2. lossfn (Required) - The loss function should be a callable with the signature lossfn(ytrue, ypred). Extensions 
                        to this argument which do not fit this standard signature paradigm will be considered later. 
            3. baselr - The base learning rate to initialize the optimizer with. Default: 1e-3. 
            3. metrics - A dictionary of metrics. The keys will be treated as the metric names. The values of the `metrics`
                        dict should be a callable with the signature metric(ytrue, ypred). 
                        For example, if you wish to track Accuracy and AUROC pass 
                        metrics = {'accuracy':accuracyfn(ytrue, ypred), 'auroc':aurocfn(ytrue, ypred)}. 
                        This will ensure that the model tracks these models under the names `accuracy` and `auroc`. 
            4. lr_scheduler_type - Pass a reference to the type of pytorch learning rate scheduler you wish to use. 
                                    For example, if you wish to use the StepLR scheduler, 
                                    set lr_scheduler_type = torch.optim.lr_scheduler.StepLR .
            5. lr_scheduler_options - A dictionary of keyword arguments to be passed to the lr_scheduler. 
                                        For example, to use the StepLR scheduler, you need to set a step_size and 
                                        multiplicative factor `gamma`. Use this argument to pass this information as
                                        a dictionary - lr_scheduler_options = {'step_size':1, 'gamma':0.5}. 
            6. optimizer_kwargs - Dictionary of additional parameters to pass to the optimizer. 
        """
        # checks 
        assert callable(lossfn)
        assert issubclass(torch.optim.Adam, torch.optim.Optimizer), 'Unrecognized optimizer type. Pass a torch optimizer type.'
        if lr_scheduler_type:
            assert issubclass(torch.optim.lr_scheduler.StepLR, torch.optim.lr_scheduler._LRScheduler),\
            'Unrecognized learning rate scheduler type. Pass a torch learning rate scheduler.'
        assert type(metrics) is dict, 'Metrics should be passed as a dict.'
        if lr_scheduler_options:
            type(lr_scheduler_options) is dict, 'Additional learning rate scheduler parameters should be a dictionary.'
        
        # initialize
        self.optimizer = optimizer(params=self.parameters(), lr=baselr, **optimizer_kwargs)
        if lr_scheduler_type:
            self.lr_scheduler = lr_scheduler(optimizer, **lr_scheduler_options)
        self.lossfn = lossfn
        self.metrics = nn.ModuleDict(metrics)
    
    def training_step(self, batch, batch_idx):
        """Whatever happens in a single training step goes here."""
        
        # the forward pass 
        xbatch, ybatch = batch
        ypred = self(xbatch)
        loss = self.lossfn(ypred, ybatch.long())
        
        # log the training loss at every iteration and the end of every epoch
        self.log(name="train_loss", value=loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        
        # calculate the metrics 
        metrics = self.metrics
        names = self.metric_names 
        for name, metric in six.iteritems(self.metrics):
            self.log(name="train_"+name, value=metric(ypred, ybatch), on_epoch=True, on_step=True, logger=True, prog_bar=True)
        return {'loss':loss}
    
    def validation_step(self, batch, batch_idx):
        """Whatever happens in a single validation step goes here."""
        
        # the forward pass in eval mode 
        xbatch, ybatch = batch
        ypred = self(xbatch)
        loss = self.lossfn(ypred, ybatch.long())
        
        # log the results 
        self.log(name="val_loss", value=loss, on_epoch=True, on_step=False, logger=True, prog_bar=True)
        for name, metric in six.iteritems(self.metrics):
            self.log(name="val_"+name, value=metric(ypred, ybatch), on_epoch=True, on_step=False, prog_bar=True)
        return {'loss':loss}
        
    def test_step(self, batch, batch_idx):
        """Whatever happens in a single test step goes here"""
        
        # the forward pass in test mode 
        xbatch, ybatch = batch
        ypred = self(xbatch)
        loss = self.lossfn(ypred, ybatch.long())
        res = {'test_loss':loss}
        
        # calculate the metrics and update return dictionary
        metrics = self.metrics
        for name, metric in six.iteritems(self.metrics):
            res['test_'+name] = metric(ypred, ybatch)
        return res
    
    def predict(self, x, return_numpy=True):
        """Make predictions at new x ; be mindful of model mode."""
        x = torch.tensor(x, dtype=self.dtype).to(self.device)
        current_mode = self.mode
        self = self.eval()
        ypred = self.forward(x)
        if current_mode == "train":
            self = self.train()
        if return_numpy:
            return ypred.cpu().data.numpy()
        else:
            return ypred
    
    def evaluate(self, xtest, ytest, verbose=False):
        """make predictions """
        
        # create a test data loader 
        x, y = torch.tensor(xtest, dtype=torch.float32), \
                            torch.tensor(ytest, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(x, y)
        batch_size = self._trainer.val_dataloaders[0].batch_size
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        res = self._trainer.test(self, dataloader, verbose=verbose)[0]
        return res
    
    def fit(self, x, y, 
            batch_size, 
            epochs=1, 
            verbose=1,
            callbacks=None,
            logger=None,
            validation_split=0.2, 
            validation_data=None, 
            validation_batch_size=None, 
            use_gpus=True,
            **kwargs):
        """
        x -> The training inputs (pytorch tensor or numpy array) 
        y -> The training outputs (pytorch tensor or numpy array)
        epochs -> Maximum number of epochs to train for.
        verbose -> Display training progress (default: 1). 
        callbacks -> A list of callbacks (pytorch_lightning.callbacks.Callback objects)
        logger -> pytorch_lightning logger to use (see here: https://pytorch-lightning.readthedocs.io/en/latest/logging.html#logging)
        validation_split -> fraction of the supplied data to use for validation.
        validation_data -> A tuple in the format (x_val, y_val) which represents validation data.
        					If supplied, will override the validation split argument. 
		validation_batch_size -> Defaults to the training batch size. 
		use_gpus -> Whether to use gpus for training. If True, training carried over all detectable GPUs.
		kwargs -> Keyword arguments passed to the pytorch_lightning trainer. 
        """

        # setup the training and validation splits 
        if validation_split and not validation_data:
            assert validation_split > 0. and validation_split < 1, "Validation split must be in (0, 1)"
            N = len(x)
            Nval = int(validation_split * N)
            allidx = np.random.permutation(np.arange(N))
            validx, trainidx = allidx[:Nval], allidx[Nval:]
            x_val, y_val = x[validx], y[validx]
            x_train, y_train = x[trainidx], y[trainidx]
        else:
            x_train, y_train = x, y
            x_val, y_val = validation_data                  
        
        # set up training and validation datasets and dataloaders 
        x_train, y_train = torch.tensor(x_train, dtype=torch.float32), \
                            torch.tensor(y_train, dtype=torch.float32)
        train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
        
        x_val, y_val = torch.tensor(x_val, dtype=torch.float32), \
                        torch.tensor(y_val, dtype=torch.float32)
        val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
        if validation_batch_size is None:
            validation_batch_size = batch_size
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=validation_batch_size)
        #return val_dataloader
        
        # instantiate a trainer 
        # under the hood the trainer puts the model into 
        # train mode. no need for manual mode change. 
        if use_gpus:
            if torch.cuda.is_available():
                gpus = torch.cuda.device_count()
        self._trainer = pl.Trainer(callbacks=callbacks, gpus=gpus, max_epochs=epochs, logger=logger)  
        self._trainer.fit(model=self,
                       train_dataloader=train_dataloader,
                       val_dataloaders=[val_dataloader])