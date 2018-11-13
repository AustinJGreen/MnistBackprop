% Simple Character recognition

% -1 = white
% 1= dark
digit0 = [ -1 1 1 1 -1 1 -1 -1 -1 1 1 -1 -1 -1 1 1 -1 -1 -1 1 1 -1 -1 -1 1 -1 1 1 1 -1 ];
digit1 = [ -1 1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 ];
digit2 = [ 1 1 1 -1 -1 -1 -1 -1 1 -1 -1 -1 -1 1 -1 -1 1 1 -1 -1 -1 1 -1 -1 -1 -1 1 1 1 1 ];

dMatrix = [ digit0 ; digit1 ; digit2 ];

t0 = [ 1 0 0 ];
t1 = [ 0 1 0 ];
t2 = [ 0 0 1 ];

tMatrix = [ t0 ; t1 ; t2 ];

learningRate = 0.01;
hiddenUnits = 240;
setEpochs = 10000;
pixelChangeCount = [ 0, 4, 8 ];

epochData = [ setEpochs setEpochs setEpochs ];
errorData = zeros(length(pixelChangeCount), setEpochs);

linePatterns = [ 'b-', 'r-', 'g-' ];
increments = [ 1 10 100 ];
hold on;

for i= 1:length(pixelChangeCount)
    
    %Get current shuffle count
    shuffleCount = pixelChangeCount(i);
    
    % Create new network
    layer1 = Layer(30, hiddenUnits, Logsig);
    layer2 = Layer(layer1, 3, Logsig);
    network = Network([layer1, layer2]);
    trainer = Trainer(network, learningRate);
    mse = 0;  
    
    for e = 1:setEpochs       
        inputMatrix = dMatrix;
        
        %Shuffle inputs
        for s = 1:shuffleCount
            for d = 1:3
                inputMatrix(d,:) = noisyNum(inputMatrix(d,:), s);
            end
        end
        
        mse = trainer.trainAll(inputMatrix, tMatrix);    
        errorData(i, e) = mse;
        
        if (mse <= 0.01)
            epochData(i) = e;
            break;
        end
        
    end
end

hold on;
for p = 1:3
    xData = 1:increments(p):epochData(p);
    yData = errorData(p,1:increments(p):epochData(p));
    plot(xData,yData,linePatterns(p));
end
hold off;

