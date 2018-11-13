classdef Network < handle

    properties
        Layers
        LayerCount
    end
    
    methods
        function network = Network(layers)    
            network.LayerCount = length(layers);
            network.Layers = layers;
            for i = 1:network.LayerCount
                layers(i).attach(network, i);
            end
        end
        
        function output = isLastLayer(network, layer)
            output = layer.Index == network.LayerCount;
        end
        
        function output = forward(network, inputs)
            output = network.Layers(1).forward(inputs);
            for i = 2:network.LayerCount
                output = network.Layers(i).forward(output);
            end
        end
        
        function backward(network, targets)
            for i = network.LayerCount:-1:1
                network.Layers(i).backward(targets);
            end
        end
        
        function update(network, alpha)
            for i = network.LayerCount:-1:1
                network.Layers(i).update(alpha);
            end          
        end
    end
end

