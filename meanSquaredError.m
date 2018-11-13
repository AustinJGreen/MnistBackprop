function [output] = meanSquaredError(errors)
output = mean(errors .* errors);
end

