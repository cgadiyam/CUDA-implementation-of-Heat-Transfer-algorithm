function SaveIterationStep( heat_array, time_step, maxTemp )
%SaveIterationStep Generates a mesh of the heat array and saves
%the image to a png file.

% Open the figure
hndl = figure();
set(hndl, 'Name', 'Heat Array Time Step');
set(gcf, 'color', [1 1 1]);

% Format the file name
number = num2str(time_step);
if( time_step < 10 )
    number = strcat('000', number);
elseif( time_step < 100 )
    number = strcat('00', number);
elseif( time_step < 1000 )
    number = strcat('0', number);
end
filename = strcat('image', number,'.png');

% Plot the data and save it to the file.
mesh(heat_array);
view(0,90);
colorbar;
caxis([0 maxTemp])
print(hndl, '-dpng', filename);

% Close the figure windows
close gcf

end