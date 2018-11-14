classdef Purelin < Tf
    methods
        function output = eval(~,x)
            output = x;
        end
        
        function output = deriv(~,~)
            output = 1;
        end
    end
end

