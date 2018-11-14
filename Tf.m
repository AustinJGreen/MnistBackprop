classdef Tf
    properties
        Func
        Deriv
    end
    
    methods
        function output = eval(~, ~)
            output = 0;
        end
        
        function output = deriv(~, ~)
            output = nan;
        end
    end
end

