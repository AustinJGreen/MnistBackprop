classdef Softmax < Tf
    methods
        function output = eval(~,x)
            e = exp(x - max(x));
            output = e / sum(e);
        end
        
        function output = deriv(~,~)
            output = 1;
        end
    end
end

