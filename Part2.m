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

hiddenUnits = [ ];
accuracies = [ ];

%Create network
layer1 = Layer(784,97,Tansig,-0.01,0.01);
layer2 = Layer(97,11,Tansig,-0.01,0.01);
layer3 = Layer(11,10,Softmax,-0.01,0.01);
network = Network([layer1, layer2, layer3]);
trainer = Trainer(network, 0.009);

for epoch = 1:35   
    idx = randperm(batchSize);
    shuffledInputs = trainImages(idx,:);
    shuffledTargets = trainLabels(idx,:);

    for b = 1:batchSize
        input = shuffledInputs(b,:);
        target = shuffledTargets(b,:);  
        actual = network.forward(input);
        network.backward(target);
        network.update(trainer.LearningRate);       
        %mse = mean((target - actual) .* (target - actual)); 
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
    disp(strcat("Epoch = ", num2str(epoch), " acc = ", num2str(accuracy)));
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
disp(strcat("Final Accuracy = ", num2str(accuracy)));