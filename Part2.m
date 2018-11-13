%Create network
layer1 = Layer(784,20,Logsig,-1,1);
%layer2 = Layer(layer1,100,Logsig,-1,1);
layer3 = Layer(layer1,10,Softmax,-1,1);
network = Network([layer1, layer3]);
trainer = Trainer(network, 0.0001);

% Fetch directory
currentFolder = pwd;

% Read mnist files
trainImagesFile = strcat(currentFolder, "\MNIST\train-images.idx3-ubyte");
trainLabelsFile = strcat(currentFolder, "\MNIST\train-labels.idx1-ubyte");
testImagesFile = strcat(currentFolder, "\MNIST\t10k-images.idx3-ubyte");
testLabelsFile = strcat(currentFolder, "\MNIST\t10k-labels.idx1-ubyte");

disp("Loading MNIST data...");
trainImages = transpose(loadMNISTImages(trainImagesFile));
trainLabels = loadMNISTLabels(trainLabelsFile);
testImages = transpose(loadMNISTImages(testImagesFile));
testLabels = loadMNISTLabels(testLabelsFile);


%Convert 28x28 -> flatten -> 784x1
%Convert labels to 10x1 array
trainLabels = convertLabel(trainLabels);

% Start training on training images
batchSize = size(trainImages, 1);

iterations = [ ];
meanSquaredErrors = [ ];

for epoch = 1:5   

    idx = randperm(batchSize);
    shuffledInputs = trainImages(idx,:);
    shuffledTargets = trainLabels(idx,:);
    
    for b = 1:batchSize
        input = shuffledInputs(b,:);
        target = shuffledTargets(b,:);  
        actual = network.forward(input);
        network.backward(target);
        network.update(trainer.LearningRate);       
        mse = meanSquaredError(target - actual); 
        
        iteration = ((epoch - 1) * batchSize) + b;
        if (mod(iteration, 1000) == 0)
            disp(strcat("Iteration ", num2str(iteration), " mse = ", num2str(mse)));
            %iterations = [ iterations, iteration ];
            %meanSquaredErrors = [ meanSquaredErrors, mse ];
            %plot(iterations, meanSquaredErrors);
            %drawnow limitrate;
        end
    end    
end

% Validate results
correctCnt = 0;
for tImage = 1:10000   
    t = testImages(tImage,:);
    output = network.forward(t);
    
    % get highest value
    prediction = getNumber(output);
    actual = testLabels(tImage);
    if (prediction == actual)
        correctCnt = correctCnt + 1;
    end
end

accuracy = (correctCnt / 10000.0) * 100;
disp(strcat("Accuracy = ", num2str(accuracy)));