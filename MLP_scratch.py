
#%%
import numpy as np

#%%
class layer:
    def __init__(self,input_size):
        self.input=None
        self.output=None
    def feedforward(self):
        raise NotImplementedError
    def backpropagation(self):
        raise NotImplementedError


class NeuronLayer(layer):
    def __init__(self,input_size,output_size):
        self.W=np.random.randn(input_size,output_size)-0.5
        self.B=np.random.randn(1,output_size)-0.5


    def feedforward(self,input):
        self.input=input
        self.output=np.dot(self.input,self.W)+self.B
        return self.output

    def backpropagation(self,error,l_r=0.1):
        input_err=np.dot(error,self.W.T)
        weight_grad=np.dot(self.input.T,error)

        ##update parameters
        self.W-= l_r*weight_grad
        self.B-= l_r * error

        return input_err


class activationLayer(layer):
    def __init__(self, func,func_der):
        self.actv_func=func
        self.actv_func_der=func_der

    
        
    def feedforward(self,input):
        self.input=input
        self.output=self.actv_func(self.input  )
        return self.output

    def backpropagation(self,output_err,l_r):
        return self.actv_func_der(self.input)*output_err


class Network:

    def __init__(self):
        self.layers=[]
    
    def mse(self,y_act,y_pred):
        return np.mean(np.power(y_act-y_pred,2))

    def der_mse(self,y_act,y_pred):
        return 2*(y_pred-y_act)/ y_act.size
    
    def addlayer(self,layer):
        self.layers.append(layer)

    def fit(self,X_train,Y_train,epoch=100,l_r=0.2):

        for _ in range(epoch):
            error=0

            for j,sample in enumerate(X_train):
                # print(sample.shape)
                # if(j==20000):
                #     break
                output=sample
                for l in self.layers:## feedforward
                    output=l.feedforward(output)
                # print("output",output[0])
                # print(Y_train[i])
                # break

                
                error +=self.mse(np.array(Y_train[j]),output)

                
                
                err=self.der_mse(np.array(Y_train[j]),output)
        
                for layer in reversed(self.layers): ## backpropagation
                    # print("err",err.shape)
                    err=layer.backpropagation(err,l_r)

            ##Average errror on all samples

            error =error/len(X_train)

            print(" Epochs {}/{} , error = {}".format(_,epoch,error))
    
    def predict(self,input):
        output_arr=[]
        for i,s in enumerate(input):
            output=input[i]
            for l in self.layers:
                output=l.feedforward(output)

            output_arr.append(output)
        
        return output_arr[0]

    
#%% DATA

train_mnist=np.genfromtxt("F:/pdf e-reading/Mtech/Datasets/mnist_train.csv",delimiter=",",skip_header=1)
test_mnist=np.genfromtxt("F:/pdf e-reading/Mtech/Datasets/mnist_test.csv",delimiter=",",skip_header=1)

x_train=train_mnist[:,1:]
x_train=x_train.reshape(x_train.shape[0],1,784)
y_train=train_mnist[:,0]
y_train1=[]
for y in y_train:
    ls=[0,0,0,0,0,0,0,0,0,0]
    ls[int(y)]=1
    y_train1.append(ls)


x_test=test_mnist[:,1:]
y_test=test_mnist[:,0]



#%% Activation functions
def sigmoid(x):
     return 1/(1+np.exp(-x))

def der_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum()

def der_softmax(softmax):
    # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
    s = softmax.reshape(-1,1)
    res=np.diagflat(s) - np.dot(s, s.T)
    print(res)
    return res



#%%Build model

def model():
    net=Network()
    net.addlayer(NeuronLayer(28*28,128))
    net.addlayer(activationLayer(sigmoid,der_sigmoid))
    net.addlayer(NeuronLayer(128,32))
    net.addlayer(activationLayer(sigmoid,der_sigmoid))
    net.addlayer(NeuronLayer(32,10))
    # net.addlayer(activationLayer(softmax,der_softmax))
    net.addlayer(activationLayer(sigmoid,der_sigmoid))
    

    return net


mymodel=model()

mymodel.fit(x_train,y_train1)





