import numpy as np

RELU = 'RelU'
SOFTMAX = 'Softmax'
LINEAR = 'Linear'
SIGMOID = 'Sigmoid'

class Layer(object):
    def __init__(self, layer_name, layer_dims=1, activation_function=RELU):
        self.layer_name = layer_name
        self.layer_dims = layer_dims
        self.activation_function = activation_function
# end class


### Softmax Regression ###

# Initialize parameters for NN for each Layer object in Ldims
def initialize_parameters(Ldims):
    parameters = {}
    TL = len(Ldims)
    for l in range(1, TL):
        parameters[f'W{l}'] = np.random.randn(Ldims[l].layer_dims, Ldims[l - 1].layer_dims) * .01
        parameters[f'b{l}'] = np.zeros((Ldims[l].layer_dims, 1))
        parameters[f'layer_obj-{l}'] = Ldims[l]

    return parameters


def _dRelU(Ab, cache):
    Z = cache
    res = (1. * (Z >= 0)).reshape(Z.shape)
    return Ab * res

def _relU(Z):
    return np.multiply(Z, Z > 0), Z

def sigmoid(Z):
    return 1/(1+ np.exp(-1*Z)), Z

def _dSigmoid(Ab, cache):
    Z = cache
    ss, _ = sigmoid(Z)
    return Ab * (ss * (1 - ss))

def softmax(Z):
    # Input tensor containing [z1....zn]
    s = np.max(Z)
    ez = np.exp(Z - s)
    return ez / np.sum(ez, axis=0, keepdims=True) # broadcasts sum over each e^z

## Forward propagation
# F prop with basic linear regression        
def _lin_f_prop(A, W, b):
    Z = W.dot(A) + b
    return Z, (A,W,b)

# F prop with activation
def lin_af(Aprev, W, b, act=RELU):
    Z, lc = _lin_f_prop(Aprev, W, b)
    if act == RELU:
        A, ac = _relU(Z)
    elif act == LINEAR:
        A, ac = Z, Z      
    else:
        raise ValueError("Could not determine activation func: {}".format(act))
    
    return A, (lc, ac)

def L_model_f_prop(X, parameters):
    # perform X-> [LDims_i]->Yhat
    Ldims = len(parameters) // 3 # W,b,Layer for each dimension
    A = X
    caches = []
    # Apply forward prop on each layer 
    for l in range(1, Ldims+1):
        A, cache = lin_af(A, 
                          parameters[f'W{l}'], 
                          parameters[f'b{l}'], 
                          parameters[f'layer_obj-{l}'].activation_function
                        )
        caches.append(cache)
    return A, caches

## Loss function
def compute_crossentropy_loss(Z, Y):
    eZ = softmax(Z)
    # Y is one hot encoded (N x K), k = num categories
    res = np.multiply(eZ, Y.T)
    total = res.sum(axis=1).sum(axis=0) # sum each category together, then all categories for average

    return -1 * total/Y.shape[0]

def compute_softmax_prediction(Z):
    eZ = softmax(Z)
    return np.argmax(eZ, axis = 0)

## Back propagation
def _lin_b_prop(dZ, cache):
    Aprev, W, b = cache
    n = Aprev.shape[1]

    dW = dZ.dot(Aprev.T) / n
    db = np.sum(dZ, axis=1, keepdims=True) / n
    dA_prev = W.T.dot(dZ)

    return dA_prev, dW, db

def lin_act_b_prop(dA, cache, act):
    # bprop with activation d/dx
    linear_cache, activation_cache = cache
    dZ = None
    if act == RELU:
        dZ = _dRelU(dA, activation_cache)
    elif act == SIGMOID:
        dZ = _dSigmoid(dA, activation_cache)    
        pass
    elif act == LINEAR:
        dZ = dA
    else:
        raise ValueError("Could not determine activation func: {}".format(act))
    
    dA_prev, dW, db = _lin_b_prop(dZ, linear_cache)
    return dA_prev, dW, db

def L_model_b_prop(AL, Y, caches):
    Ldims = len(caches)

    grads = {}
    cur_cache = caches[Ldims - 1]
    # dAL = dL/dA * dA/dz (softmax)
    dAL = AL-Y.T # Y is 1-hot encoded for the relevent category
    dA_prt, dW_t, db_t = lin_act_b_prop(dAL, cur_cache, "Linear")
    grads["dA" + str(Ldims-1)] = dA_prt
    grads["dW" + str(Ldims)] = dW_t 
    grads["db" + str(Ldims)] = db_t

    for l in reversed(range(Ldims-1)):
        current_cache = caches[l]
        dA_prt, dW_t, db_t = lin_act_b_prop(grads["dA" + str(l + 1)], current_cache, "Sigmoid")
        grads["dA" + str(l)] = dA_prt
        grads["dW" + str(l + 1)] = dW_t 
        grads["db" + str(l + 1)] = db_t

    return grads

def update_params(params, grads, learning_rate):
    params = params.copy()
    Ldims =  len(params) // 3
    for l in range(Ldims):
        params["W" + str(l+1)] = params["W" + str(l+1)] - learning_rate*grads["dW" + str(l+1)]
        params["b" + str(l+1)] = params["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)]
    return params

def predict(params, input):
    Z, _ = L_model_f_prop(input, params)
    return compute_softmax_prediction(Z)
