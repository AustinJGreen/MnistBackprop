classdef Tansig < Tf
    methods
        function output = eval(tf,x)
            expN = exp(x);
            expNN = exp(-x);
            numerator = expN - expNN;
            denominator = expN + expNN;
            output = numerator ./ denominator; % ./ is elementwise division
        end
        
        function output = deriv(tf,y)
            output = 1 - (y .* y);
        end
    end
end

