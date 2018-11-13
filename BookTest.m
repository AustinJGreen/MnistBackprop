layer1 = Layer(1, 2, Logsig, 0, 1);
layer1.Weights = [ -0.27 ; -0.41 ];
layer1.Biases = [ -0.48 ; -0.13 ];
layer2 = Layer(2, 1, Purelin, 0, 1);
layer2.Weights = [ 0.09 -0.17 ];
layer2.Biases = [ 0.48 ];
net = Network([layer1, layer2]);
output = net.forward(1);
net.backward(1.707106);
% Page 11-16 ish