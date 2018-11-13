classdef Tansig < Tf
    methods
        function output = eval(tf,x)
            expN = exp(x);
            expNN = exp(-x);
            numerator = expN - expNN;
            denominator = expN + expNN;
            output = numerator ./ denominator; % ./ is elementwise division
        end
        
        function output = deriv(tf,x)
            expn = exp(2 .* x);
            numerator = 4 .* expn;
            denominator = (expn + 1) .* (expn + 1);
            output = numerator ./ denominator;
        end
    end
end

