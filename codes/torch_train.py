import torch
import numpy as np


class TorchTrain:
    """A class for training a model in PyTorch.

    Parameters
    -----------
        model (torch.nn.Module): The PyTorch model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        loss_function (callable): The loss function to use for training.
        metrics (dict or callable, optional): The metrics to evaluate during training.
            If a dictionary, the keys are the metric names and the values are functions that
            take in `yhat` and `y` and return a metric value. If a callable, it should take
            in `yhat` and `y` and return a metric value. Defaults to None.

    Attributes
    -----------
        DEVICE (torch.device): The device to use for training (cuda if available, cpu otherwise).
        model (torch.nn.Module): The PyTorch model being trained.
        optimizer (torch.optim.Optimizer): The optimizer being used for training.
        loss_function (callable): The loss function being used for training.
        metrics (dict or callable): The metrics being evaluated during training.
        metrics_evaluated (dict): The metrics evaluated during training.
        train_loss (float): The average training loss.
        test_loss (float): The average test loss.
        train_iteration (int): The number of training iterations.
        test_iteration (int): The number of test iterations.
        train_metrics (dict): The metrics evaluated on the training data.
        test_metrics (dict): The metrics evaluated on the test data.
    """

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(
        self, model, optimizer, loss_function, metrics=None, scheduler=None
    ) -> None:
        """Initialize the TorchTrain object.

        Parameters
        -----------
        model : torch.nn.Module
            The PyTorch model to train.
        optimizer : torch.optim.Optimizer
            The optimizer to use for training.
        loss_function : callable
            The loss function to use for training.
        metrics : dict or callable, optional
            The metrics to evaluate during training. If a dictionary, the keys are the metric names
            and the values are functions that take in `yhat` and `y` and return a metric value.
            If a callable, it should take in `yhat` and `y` and return a metric value. Defaults to None.
        scheduler : torch.optim.lr_scheduler, optional
            The learning rate scheduler to use for training. Defaults to None.
        """
        self.model = model
        self.model.to(self.DEVICE)
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metrics = self.__preprocess_metrics(metrics)
        self.scheduler = scheduler
        self.metrics_evaluated = {}
        self.train_loss = 0
        self.test_loss = 0
        self.train_iteration = 0
        self.test_iteration = 0
        self.train_metrics = {}
        self.test_metrics = {}
        self.history = {}
        self.train_loss_all = []
        self.test_loss_all = []
        self.train_metrics_all = []
        self.test_metrics_all = []
        self.__train_scaled = False
        self.__test_scaled = False

    def __preprocess_metrics(self, metrics):
        """Preprocesses the given metrics"""
        if metrics is None:
            return {}
        if isinstance(metrics, dict):
            return {key.title(): value for key, value in metrics.items()}
        else:
            raise TypeError(
                "Metrics should be a dictionary of metrics or a function which takes yhat, y"
            )

    def __scale_matrices(self, loss, metrics, type="train"):
        """Scales the loss and metrics

        Parameters
        -----------
        loss : float
            The loss to scale
        metrics : dict
            The metrics to scale
        type : str, optional
            The type of scaling to do, either "train" or "test", by default "train"

        Returns
        --------
        loss : float
            The scaled loss
        metrics : dict
            The scaled metrics
        """
        if type == "train" and not self.__train_scaled:
            scale = self.train_iteration
            self.__train_scaled = True
        elif type == "test" and not self.__test_scaled:
            scale = self.test_iteration
            self.__test_scaled = True
        else:
            return loss, metrics
        loss /= scale
        for key in metrics:
            metrics[key] /= scale
        return loss, metrics

    def __reset_counters(self):
        """Resets all the counters and loss objects for a new epoch"""
        self.train_loss, self.train_metrics = self.__scale_matrices(
            self.train_loss, self.train_metrics, type="train"
        )

        self.test_loss, self.test_metrics = self.__scale_matrices(
            self.test_loss, self.test_metrics, type="test"
        )

        self.train_loss_all.append(self.train_loss)
        self.train_loss = 0

        self.test_loss_all.append(self.test_loss)
        self.test_loss = 0

        self.train_iteration = 0
        self.test_iteration = 0

        self.train_metrics_all.append(self.train_metrics)
        self.train_metrics = {}

        self.test_metrics_all.append(self.test_metrics)
        self.test_metrics = {}
        self.__train_scaled = False
        self.__test_scaled = False

    @property
    def loss(self):
        """Returns the training loss"""
        return self.train_loss_all[-1]

    def __create_history(self):
        """Creates the history dictionary"""
        history = {
            "train_loss": self.train_loss_all,
            "val_loss": self.test_loss_all,
        }
        for key, value in self.metrics.items():
            history[f"train_{key.lower()}"] = []
            history[f"val_{key.lower()}"] = []

        for item in self.train_metrics_all:
            for key, value in item.items():
                history[f"train_{key.lower()}"].append(value)

        for item in self.test_metrics_all:
            for key, value in item.items():
                history[f"val_{key.lower()}"].append(value)
        return history

    def __parse_val(self, val):
        """Parses the given value to a float"""
        if isinstance(val, torch.Tensor):
            val = val.item()
        elif isinstance(val, np.ndarray):
            val = float(val)
        elif isinstance(val, (int, float)):
            pass
        else:
            raise TypeError(
                f"The given Metric function should return a tensor, numpy array, int, or float.\n\
                    Got {type(val)}"
            )
        return val

    def _train_step(self, x, y):
        """Perform a single training step.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        y : torch.Tensor
            The target tensor.

        Returns
        -------
        tuple
            A tuple containing the loss and the predicted output tensor.
        """
        self.model.train()
        yhat = self.model(x)
        l = self.loss_function(yhat, y)
        self.optimizer.zero_grad()
        l.backward()
        self.optimizer.step()
        self.train_iteration += 1
        return l.item(), yhat

    def _test_step(self, x, y):
        """Perform a single testing step.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        y : torch.Tensor
            The target tensor.

        Returns
        -------
        tuple
            A tuple containing the loss and the predicted output tensor.
        """
        self.model.eval()
        with torch.inference_mode():
            yhat = self.model(x)
            l = self.loss_function(yhat, y)
            self.test_iteration += 1
        return l.item(), yhat

    def predict(self, x):
        """Make predictions on a batch of data.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The predicted output tensor.
        """
        self.model.eval()
        yhat = self.model(x)
        yhat = torch.argmax(yhat, dim=1)
        return yhat

    def __calculate_metrics(self, yhat, y):
        """Calculate the metrics for a batch of data.

        Parameters
        ----------
        yhat : torch.Tensor
            The predicted output tensor.
        y : torch.Tensor
            The target tensor.

        Returns
        -------
        dict
            A dictionary containing the values of the metrics.
        """
        metrics = {}
        for key, metric in self.metrics.items():
            val = metric(yhat, y)
            if isinstance(val, torch.Tensor):
                val = val.item()
            elif isinstance(val, np.ndarray):
                val = float(val)
            elif isinstance(val, (int, float)):
                pass
            else:
                raise TypeError(
                    f"Metric {key} should return a tensor, numpy array, int, or float"
                )
            metrics[key] = val
        self.metrics_evaluated = metrics
        return metrics

    def __progress_bar(self, cur_iter, all_iter):
        """Creates a progress bar showing the progress of the current batch.

        Parameters
        ----------
        cur_iter : int
            The current batch number.
        all_iter : int
            The total number of batches.

        Returns
        -------
        str
            The progress bar, in the form of "10/100[====----]".
        """
        len_progress_bar = 20
        progress = int((cur_iter + 1) / all_iter * len_progress_bar)
        progress_bar = "=" * progress + "-" * (len_progress_bar - progress)
        return f"[{progress_bar}]"

    def progress(self, cur_iter, all_iter, loss, metrics, on="train"):
        """Prints a progress bar showing the progress of the current batch.

        Parameters
        ----------
        cur_iter : int
            The current batch number.
        all_iter : int
            The total number of batches.
        loss : float
            The current loss. Should be averaged over all batches.
        metrics : dict
            The metrics evaluated on the current batch.
        on : str, optional
            Whether the progress bar is for the training or testing data. Defaults to "train".

        Returns
        -------
        str
            The progress bar, in the form of "10/100[====----]".

        Notes
        -----
        The progress bar shows the progress of the current batch as a bar of equal signs ("=") and
        hyphens ("-"). The length of the bar is fixed at 20 characters. The current batch number
        and total number of batches are displayed at the beginning of the progress bar. The current
        loss and any metrics evaluated on the current batch are displayed at the end of the progress
        bar.
        """
        # len_progress_bar = 20
        # progress = int((cur_iter + 1) / all_iter * len_progress_bar)
        # progress_bar = "=" * progress + "-" * (len_progress_bar - progress)
        progress_bar = self.__progress_bar(cur_iter=cur_iter, all_iter=all_iter)

        if on.lower() == "train":
            iteration = self.train_iteration
            prefix = f"Epoch {(self.current_epoch+1):2d}/{self.epochs:2d} Batch "
        else:
            iteration = self.test_iteration
            prefix = "Epoch "

        text = f"{prefix}{cur_iter:>4d}/{all_iter:>4d}{progress_bar} {on.title()} loss: {loss/iteration:.4f}"
        for metric_name, metric_value in metrics.items():
            text += f" | {on.title()} {metric_name}: {metric_value/iteration:.4f}"

        return text

    def update_metrics(self, cur_metrics, new_metrics):
        """Update the metrics with the values for a new batch of data.

        Parameters
        ----------
        cur_metrics : dict
            The current values of the metrics.
        new_metrics : dict
            The values of the metrics for a new batch of data.

        Returns
        -------
        dict
            A dictionary containing the updated values of the metrics.
        """
        for key, value in new_metrics.items():
            if key not in cur_metrics:
                cur_metrics[key] = value
            else:
                cur_metrics[key] += value
        return cur_metrics

    def fit(
        self,
        train_loader,
        validation_data_loader=None,
        epochs=1,
        verbose=True,
        train_steps_per_epoch=None,
        validation_steps_per_epoch=None,
    ):
        """Fit the PyTorch model.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            The data loader for the training data.
        validation_data_loader : torch.utils.data.DataLoader, optional
            The data loader for the test data. Defaults to None.
        epochs : int, optional
            The number of epochs to train for. Defaults to 1.
        verbose : bool, optional
            Whether to print the training progress during training. Defaults to True.
        train_steps_per_epoch : int, optional
            The number of batches to train on per epoch. Defaults to None.
        validation_steps_per_epoch : int, optional
            The number of batches to test on per epoch. Defaults to None.

        Returns
        -------
        None

        Examples
        --------
        >>> model = MyModel()
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> loss_function = nn.CrossEntropyLoss()
        >>> scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        >>> train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        >>> validation_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
        >>> trainer = TorchTrain(model, optimizer, loss_function, scheduler=scheduler)
        >>> trainer.fit(train_loader, validation_data_loader=validation_data_loader, epochs=10, verbose=True)
        """
        self.epochs = epochs
        if train_steps_per_epoch is None:
            train_steps_per_epoch = len(train_loader)
        if validation_data_loader is not None:
            if validation_steps_per_epoch is None:
                validation_steps_per_epoch = len(validation_data_loader)

        for epoch in range(epochs):
            self.current_epoch = epoch
            for i, (x, y) in enumerate(train_loader):
                x = x.to(self.DEVICE)
                if isinstance(y, list) or isinstance(y, tuple):
                    y = [y_.to(self.DEVICE) for y_ in y]
                else:
                    y = y.to(self.DEVICE)

                train_loss, yhat = self._train_step(x, y)
                self.train_loss += train_loss
                metrics = self.__calculate_metrics(yhat, y)
                self.train_metrics = self.update_metrics(self.train_metrics, metrics)

                b_progress = self.progress(
                    i + 1,
                    train_steps_per_epoch,
                    self.train_loss,
                    self.train_metrics,
                    on="train",
                )
                if i == train_steps_per_epoch - 1:
                    print(b_progress)
                    break
                else:
                    if verbose:
                        print(b_progress, end="\r")
            if validation_data_loader is not None:
                for i, (x, y) in enumerate(validation_data_loader):
                    x = x.to(self.DEVICE)
                    if isinstance(y, list) or isinstance(y, tuple):
                        y = [y_.to(self.DEVICE) for y_ in y]
                    else:
                        y = y.to(self.DEVICE)
                    test_loss, yhat = self._test_step(x, y)
                    self.test_loss += test_loss
                    metrics = self.__calculate_metrics(yhat, y)
                    self.test_metrics = self.update_metrics(self.test_metrics, metrics)
                    if i == validation_steps_per_epoch - 1:
                        break
                test_progress = self.progress(
                    epoch + 1,
                    epochs,
                    self.test_loss,
                    self.test_metrics,
                    on="test",
                )
                print(test_progress)
            self.__reset_counters()
            if self.scheduler is not None:
                self.scheduler.step()
            if verbose and self.scheduler is not None:
                print(f"New Learning rate: {self.scheduler.get_last_lr()[0]:.6f}")

        return self.__create_history()

    def save(self, path):
        """Save the model to a file.

        Parameters
        ----------
        path : str
            The path to the file to save the model to.
        """
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        """Load the model from a file.

        Parameters
        ----------
        path : str
            The path to the file to load the model from.
        """
        self.model.load_state_dict(torch.load(path))

    def evaluate(self, data_loader, metric):
        """Evaluate the model on a data loader and the given metric.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            The data loader to evaluate the model on.
        metric : function
            The metric to evaluate the model with.

        Returns
        -------
        float
            The score of the model on the given metric.
        """
        running_score = 0
        data_length = len(data_loader)
        for i, (x, y) in enumerate(data_loader):
            progress_bar = self.__progress_bar(i, data_length)
            x = x.to(self.DEVICE)
            if isinstance(y, list) or isinstance(y, tuple):
                y = [y_.to(self.DEVICE) for y_ in y]
            else:
                y = y.to(self.DEVICE)

            yhat = self.predict(x)
            score = metric(y, yhat)
            score = self.__parse_val(score)
            running_score += score

            progress_bar = f"{i+1}/{data_length}" + progress_bar
            progress_bar += f" Score: {(running_score/(i+1)):4f}"
            print(progress_bar, end="\r")
        return running_score / (len(data_loader))
