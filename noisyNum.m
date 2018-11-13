function out = noisyNum(in,count)
    n = length(in);
    i = 1;
    visitedIndex = [];
    while i <= count
        x = randsample(n, 1);
        if (ismember(x, visitedIndex) == 0)
           visitedIndex = [x visitedIndex];
           in(x) = in(x) * -1; %Flip digit color
           i = i + 1;
        end
    end
    out = in;
end

