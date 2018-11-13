% Fetch directory
currentFolder = pwd;

% Read mnist files
trainImagesFile = strcat(currentFolder, "\MNIST\train-images.idx3-ubyte");
trainLabelsFile = strcat(currentFolder, "\MNIST\train-labels.idx1-ubyte");
testImagesFile = strcat(currentFolder, "\MNIST\t10k-images.idx3-ubyte");
testLabelsFile = strcat(currentFolder, "\MNIST\t10k-labels.idx1-ubyte");
disp("Loading MNIST data...");
[trainImages, trainLabels] = readMNIST(trainImagesFile,trainLabelsFile,60000,0);
[testImages, testLabels] = readMNIST(testImagesFile,testLabelsFile,10000,0);

%Convert 28x28 -> flatten -> 784x1
%Convert labels to 10x1 array
disp ("Reshaping data...");
trainImages = reshape(trainImages, [60000, 784]);
testImages = reshape(testImages, [10000, 784]);
trainLabels = convertLabel(trainLabels);
testLabels = convertLabel(testLabels);

layer1 = Layer(784,300,Logsig);
layer2 = Layer(layer1,100,Logsig);
layer3 = Layer(layer2,10,Logsig);
network = Network([layer1, layer2, layer3]);
trainer = Trainer(network, 0.01);

% Start training on training images
for epoch = 1:10 
    mse = trainer.trainAll(trainImages,trainLabels); 
    disp(strcat("epoch = ", num2str(epoch), " mse = ", num2str(mse)));
end

% Validate results
correctCnt = 0;
for tImage = 1:10000
    output = network.forward(testImages(tImage));
    
    % get highest value
    prediction = getNumber(output);
    actual = testLabels(tImage);
    if (prediction == actual)
        correctCnt = correctCnt + 1;
    end
end

accuracy = (correctCnt / 10000.0) * 100;
disp(strcat("Accuracy = ", num2str(accuracy)));