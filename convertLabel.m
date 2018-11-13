function [result] = convertLabel(label)
    result = zeros(length(label), 10);
    for i = 1:length(label)
        labelIndex = label(i)+1;
        result(i, labelIndex) = 1;
    end
end