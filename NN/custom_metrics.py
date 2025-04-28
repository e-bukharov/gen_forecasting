from keras import ops

def rmse (y_true, y_pred):
    """
    Root mean square error. Return RMSE
    """
    loss = ops.square(y_true - y_pred) # RMSE
    return ops.sqrt(ops.mean(loss))
    
def rmsle (y_true, y_pred, const=0.0):
    """
    Root mean square log error. Return RMSLE. Constan normalizes values to prevent zero values of log. RMSLE is just used for non-negative values
    """
    loss = ops.square(ops.log(y_true + const) - ops.log(y_pred + const)) # RMSLE
    return ops.sqrt(ops.mean(loss))
    
def mape_to_max_power (y_true, y_pred):
    """
    Mean absolute precentage error but normilized by maximal value in goten datas. Return MAPE. 
    """
    loss = abs(y_true - y_pred)/ops.max(y_true)*100
    return ops.mean(loss)

def smape (y_true, y_pred, const=0.0):
    """
    Symmetric mean absolute percentage error. Return SMAPE. The metrics minimize probability of zero division. Constan prevents division by zero
    """
    loss = 2*abs(y_true - y_pred)/(abs(y_true) + abs(y_pred) + const)*100 #SMAPE
    return ops.mean(loss)

def R_2 (y_true, y_pred):
    """
    R square metrics. Return R_2
    """
    return (1 - (ops.sum(ops.square(y_true - y_pred)) / ops.sum(ops.square(y_true-ops.mean(y_true)))))