function [number] = getNumber(activation)
    maxValue = activation(1);
    number = 0;
    for i = 2:length(activation)
        curNumber = activation(i);
        if (curNumber > maxValue)
            maxValue = curNumber;
            number = i - 1;
        end
    end
end

