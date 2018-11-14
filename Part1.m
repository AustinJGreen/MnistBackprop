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
hiddenUnits = 40;
setEpochs = 5000;
pixelChangeCount = [ 0 4 8 ];
colors = [ "red", "green", "blue" ];
legendStr = [ "0 shuffles", "4 shuffles", "8 shuffles" ];

epochData = [ setEpochs setEpochs setEpochs ];
errorData = zeros(length(pixelChangeCount), setEpochs);

increments = [ 1 10 100 ];

for i= 1:length(pixelChangeCount)
    
    %Get current shuffle count
    shuffleCount = pixelChangeCount(i);
    
    % Create new network
    layer1 = Layer(30, hiddenUnits, Logsig, -1, 1);
    layer2 = Layer(layer1, 3, Logsig, -1, 1);
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
    end
end

hold on;
for p = 1:length(pixelChangeCount)
    xData = 1:increments(p):epochData(p);
    yData = errorData(p,1:increments(p):epochData(p));
    plotHandle = plot(xData,yData,'-');
    set(plotHandle, "Color", colors(p));
    title("MSE of classification compared to number of epochs");
    xlabel("Epochs");
    ylabel("Mean Squared Error");
end
legend(legendStr);
hold off;

