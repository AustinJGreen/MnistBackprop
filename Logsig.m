classdef Logsig < Tf
    methods
        function output = eval(tf,x)
            output = 1 ./ (1 + exp(-x)); % ./ is elementwise division
        end
        
        function output = deriv(tf,x)
            ex = exp(-x);
            output = ex ./ ((1 + ex) .* (1 + ex));
        end
    end
end

