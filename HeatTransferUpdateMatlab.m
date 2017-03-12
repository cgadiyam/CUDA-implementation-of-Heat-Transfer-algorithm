%********************************************************************
% Jason Lowden
% October 21, 2013
%
% The purpose of this function is to perform the heat transfer
% update step that is to be performed by Matlab.
%********************************************************************
function [ result_heat_array ] = HeatTransferUpdateMatlab( heat_array, ...
    heat_speed, iterations )
%HeatTransferUpdateMatlab Updates the heat map using the given speed
%and operates for the specified number of iterations. The output variable
%is the result of the entire function.

%Make a copy of the array to maintain the state of the input parameter
heat_array_copy = heat_array;
heat_array_updated = heat_array;
matrixSize = length(heat_array);

for i=1:iterations
    for x=2:(matrixSize-1)
        for y=2:(matrixSize-1)
            t_old = heat_array_copy(x,y);

            t_new = heat_array_copy(x-1,y) + heat_array_copy(x+1,y) + ...
                heat_array_copy(x,y-1) + heat_array_copy(x,y+1) - 4 * t_old;

            t_new = t_old + heat_speed * t_new;
            heat_array_updated(x,y) = t_new;
        end
    end

    %Copy the result into the reference matrix for the next update.
    heat_array_copy = heat_array_updated;
end

%Put the resulting matrix into the output parameter.
result_heat_array = heat_array_copy;

end