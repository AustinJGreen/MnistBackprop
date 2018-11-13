classdef Layer < handle
    %Layer Handles individual layer processing
    
    properties
        % The index of the layer in the network
        Index
        
        % The network the layer is in
        Network
        
        % The amount of inputs in a vector this layer takes in
        InputCount
        
        % The amount of neurosn in this layer
        NeuronCount
        
        % The transfer function for this layer
        Tf
        
        % The weights for this layer
        Weights
        
        % The biases for this layer
        Biases
        
        % Gradient
        Grad
        
        % The last input passed into this Layer (a - 1)
        LastInput
        
        % The last input passed into this Layer's Tf (n)
        LastNetInput
        
        % The last output passed out of this layer (a)
        LastOutput
    end
    
    methods

        function layer = Layer(arg1, neuronCount, tf, minInit, maxInit)
            if (class(arg1) == "Layer")
                layer.InputCount = arg1.NeuronCount;
            else
                layer.InputCount = arg1;
            end        

            layer.NeuronCount = neuronCount;
            
            % Create weight matrix and initialize uniformly between
            % [minInit, maxInit]
            maxDif = maxInit - minInit;
            layer.Weights = (rand(neuronCount, layer.InputCount) * maxDif) + minInit;
            
            % Create bias vector and initialize with 0s for now
            layer.Biases = zeros(neuronCount, 1);

            layer.Tf = tf;
            
            % Initialize output with nan's so its at least not null for
            % calculations
            layer.LastOutput = nan(1, neuronCount);
            
            % Initialize gradient
            layer.Grad = zeros(layer.NeuronCount, 1);
        end
        
        function attach(layer, network, index)
            layer.Network = network;
            layer.Index = index;
        end
        
        function output = forward(layer, inputs)
            layer.LastInput = inputs;
            layer.LastNetInput = layer.Weights * transpose(inputs) + layer.Biases;
            layer.LastOutput = transpose(layer.Tf.eval(layer.LastNetInput));
            output = layer.LastOutput;
        end
        
        function backward(layer, targets)
            if (layer.Network.isLastLayer(layer))
                
                % S(m) = -2tfderiv(n(m)) * (t - a)
                error = transpose((targets - layer.LastOutput));
                tfS = layer.Tf.deriv(layer.LastNetInput);
                
                % Calculate gradient only based on current gradients
                layer.Grad = ((-2 * tfS) .* error);
            else   
                
                % S(m) = tfderiv(n(m)) * W(m+1)'*S(m+1)
                nextLayer = layer.Network.Layers(layer.Index + 1);
                
                % Calculate gradient only based on current gradients
                error = transpose(nextLayer.Weights) * nextLayer.Grad;  
                
                % Calculate current gradient, and update mean
                layer.Grad = layer.Tf.deriv(layer.LastNetInput) .* error;               
            end
        end
        
        function update(layer, alpha)            
            % Use  grad to update
            layer.Weights = layer.Weights - (alpha * layer.Grad) * layer.LastInput;
            layer.Biases = layer.Biases - (alpha * layer.Grad);
        end
    end
end

