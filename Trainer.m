classdef Trainer < handle

    properties
        Y
        X
        Network
        LearningRate
    end
    
    methods
        function trainer = Trainer(network, learningRate)
            trainer.Network = network;
            trainer.LearningRate = learningRate;
        end
        
        function mse = train(trainer, input, target)
            actual = trainer.Network.forward(input);
            trainer.Network.backward(target);
            trainer.Network.update(trainer.LearningRate);
            mse = mean((target - actual) .* (target - actual));
        end
        
        function mse = trainAll(trainer, inputs, targets)
            batchSize = size(inputs, 1);
            targetBatchSize = size(targets, 1);
            assert(batchSize == targetBatchSize);
            
            % Shuffle batch
            idx = randperm(batchSize);
            shuffledInputs = inputs(idx,:);
            shuffledTargets = targets(idx,:);
            
            sum = 0;
            for b = 1:batchSize
                target = shuffledTargets(b,:);
                input = shuffledInputs(b,:);
                actual = trainer.Network.forward(input);
                trainer.Network.backward(target);
                trainer.Network.update(trainer.LearningRate);
                sum = sum + ((target - actual) .* (target - actual)); 
            end

            mse = mean(sum / batchSize);
        end
    end
end

