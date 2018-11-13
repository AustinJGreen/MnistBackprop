classdef Purelin < Tf
    methods
        function output = eval(tf,x)
            output = x;
        end
        
        function output = deriv(tf,x)
            output = 1;
        end
    end
end

