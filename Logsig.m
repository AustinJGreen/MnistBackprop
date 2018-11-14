classdef Logsig < Tf
    methods
        function output = eval(~,x)
            output = 1 ./ (1 + exp(-x)); % ./ is elementwise division
        end
        
        function output = deriv(~,y)
            output = y .* (1 - y);
        end
    end
end

