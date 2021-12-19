import tensorflow as tf
from kerasncp import wirings
from kerasncp.tf import LTCCell

class SaveBestModel(tf.keras.callbacks.Callback):
    def __init__(self, save_best_metric='val_root_mean_squared_error'):
        self.save_best_metric = save_best_metric
        self.lowestError = 470

    def on_epoch_end(self, epoch, logs=None):
        # print(logs[self.save_best_metric])
        if (logs[self.save_best_metric] < self.lowestError):
            self.lowestError = logs[self.save_best_metric]
            # self.model.save(f"supply_model/{seq_len}hrinputLiquid-{round(logs[self.save_best_metric],2)}-RMSE.model")
            self.model.save(
                f"supply_model_v2/Liquid-{round(logs[self.save_best_metric], 2)}-RMSE.model")
            print(f"\n------Model with RMSE of {logs[self.save_best_metric]} saved------")