classdef Softmax < Tf
    methods
        function output = eval(tf,x)
            e = exp(x - max(x));
            output = e / sum(e);
        end
        
        function output = deriv(tf,y)
            output = 1;
        end
    end
end

