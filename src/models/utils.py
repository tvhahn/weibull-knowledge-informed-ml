import numpy as np
import torch
import torch.nn as nn
from src.models.loss import RMSELoss, RMSLELoss
from sklearn.metrics import r2_score
import pandas as pd


#########################
# EARLY STOPPING
#########################


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0.005,
        path="checkpoint.pt",
        trace_func=print,
        early_stop_delay=20,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print      

        From https://github.com/Bjarten/early-stopping-pytorch
        License: MIT     
        """
        self.patience = patience
        self.verbose = verbose
        self.early_stop_delay = early_stop_delay
        self.counter = 0
        self.epoch = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss
        # print(type(score), 'SCORE ####,', score)

        if self.epoch < self.early_stop_delay:
            self.epoch += 1
            pass
        else:
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                if self.verbose:
                    self.trace_func(
                        f"EarlyStopping counter: {self.counter} out of {self.patience}"
                    )
                if self.counter >= self.patience:
                    self.early_stop = True
            elif torch.isnan(score).item():
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                # print('########## IS NAN #######')
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0
            self.epoch += 1

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model, self.path)
        # torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

#########################
# TESTING
#########################


def test(net, x_test, device, batch_size=100, ):
    with torch.no_grad():  

        y_hats = []
        
        for i in range(0, len(x_test), batch_size):
            batch_x = x_test[i:i+batch_size].to(device)
            outputs = net(batch_x)
            y_hats.append(np.array(outputs.cpu()).reshape(-1,1))
    
    return torch.tensor(np.concatenate(y_hats))


def calc_r2_avg(y_hats, y_val, index_sorted, window_size):
    y_hats_rolling_avg = np.convolve(np.array(y_hats[index_sorted]).reshape(-1), np.ones(window_size), 'valid') / window_size
    r2_val_avg = r2_score(np.array(y_val)[index_sorted][window_size-1:], y_hats_rolling_avg)
    return r2_val_avg, y_hats_rolling_avg

# function to create metrics from the test set on an already trained model
def model_metrics_test(net, model_path, x_test, y_test, device,  window_size=12):
    net.eval()
    
    criterion_mae = nn.L1Loss()
    criterion_rmse = RMSELoss()
    criterion_rmsle = RMSLELoss()
    
    results_dict = {}

    try:
        y_hats = test(net, x_test, device, 100)
        index_sorted = np.array(np.argsort(y_test, 0).reshape(-1))
        
        r2_test = r2_score(y_test, y_hats)
        results_dict['r2_test'] = r2_test
        r2_test_avg, y_hats_rolling_avg = calc_r2_avg(y_hats, y_test, index_sorted, window_size)
        results_dict['r2_test_avg'] = r2_test_avg
        loss_mae_test = criterion_mae(y_hats, y_test)
        results_dict['loss_mae_test'] = loss_mae_test.item()
        loss_rmse_test = criterion_rmse(y_hats, y_test)
        results_dict['loss_rmse_test'] = loss_rmse_test.item()
        loss_rmsle_test = criterion_rmsle(y_hats, y_test)
        results_dict['loss_rmsle_test'] = loss_rmsle_test.item()
    except:
        results_dict['r2_test'] = 99999
        results_dict['r2_test_avg'] = 99999
        results_dict['loss_mae_test'] = 99999
        results_dict['loss_rmse_test'] = 99999
        results_dict['loss_rmsle_test'] = 99999
    
    return results_dict


def test_metrics_to_results_df(model_folder, df_results, x_test, y_test, ):
    '''Function that takes the results datafram and appends
    the results from the test data to it.
    
    Parameters
    ===========
    model_folder : pathlib position 
        Folder holding all the saved checkpoint files of saved models  
    '''

    # select device to run neural net on
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on GPU")
    else:
        device = torch.device("cpu")
        print("Running on CPU")

    df_temp = pd.DataFrame()

    for i, r in df_results.iterrows():
        model_name = r['model_checkpoint_name']

        if i % 200 == 0:
            print('model no. ', i)

        # load model 
        net = torch.load(model_folder / model_name, map_location=device)

        results_dict = model_metrics_test(net, model_folder / model_name, x_test, y_test, device)
        results_dict['model_name'] = model_name

        df_temp = df_temp.append(pd.DataFrame.from_dict(results_dict,orient='index').T)

    df_temp = df_temp.reset_index(drop=True)
        
    return df_results.join(df_temp.drop(columns='model_name'))
