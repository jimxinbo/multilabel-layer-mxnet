# multilabel-layer-mxnet


# what is it. 
This is a mxnet operator layer for multilabel classification. It implements multilabel softmax.
We assume the num of labels for each sample is known and fixed, denoted as k. Then the ground truth label matrix (size nxk where n is the num of sampels per batch) have values of [0,K-1] where K is the largest label index.
(In the case k is not fixed for each sample, one can bulid the ground truth label matrix of size nxkm, where km is the largest possible num of labels for any sample, and assign the useless entries to be any values outside [0, K-1])

# how to use it.
Put the 3 files (softmax_multilabel_output-inh.h, softmax_multilabel_output.cc, softmax_multilabel_output.cu) into mxnet/src/operator and recompile mxnet.
Then one can use it as any other operators. The following is a simple example.

    ...
    num_label = 3
    X = mx.sym.Variable("data")
    fc = mx.sym.FullyConnected(data=net, num_hidden=100, name="fc")
    msoftmax = mx.sym.SoftmaxMultilabelOutput(data=fc, name='msoftmax', num_label=num_label)
    executor = msoftmax.simple_bind(ctx=ctx, data=data_shape, grad_req='write')
    ...
    
# some thoughts. 
In theory, the ground truth matrix often appears to be of size nxK with 0s and 1s. We have assumed that k is usually much smaller than K and the current choice save certain space. 


