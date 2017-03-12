%********************************************************************
% Jason Lowden
% October 21, 2013
%
% The purpose of this script is to generate the heat array that will
% be used for the CUDA heat transfer simulation and then perform the 
% CPU and GPU timings that will be used for comparison.
%********************************************************************

%Generate the data that will be used.
clear all;
close all;

%Define the size of the grid to use.
size = 1000;

%Create the matrix to represent the data
heat_array = zeros(size,size);

%Create some random hot spots in the array
numHotSpots = 23;
sizeHotSpots = floor(size / 13);
if( mod(sizeHotSpots,2) == 0 )
	sizeHotSpots = sizeHotSpots + 1;
end
maxTemperature = 1000;

%Generate the hot spots
for i=1:numHotSpots
    %Get the center coordinate to create the points at.
    centerX = ceil( rand() * size );
    centerY = ceil( rand() * size );
    
    %Get a random temperature for the point.
    temperature = rand() * maxTemperature;
    
    %Iterate over the center and create the hot spot
    for j=-floor(sizeHotSpots / 2):floor(sizeHotSpots / 2)
        newX = centerX + j;
        if( newX >= 1 && newX <= size)
            for k=-floor(sizeHotSpots / 2):floor(sizeHotSpots / 2)
                newY = centerY + k;
                if( newY >= 1 && newY <= size)
                    heat_array(newX,newY) = temperature;
                end
            end
        end
    end
end

%Run the update for some number of times.
heatSpeed = 0.2;
timeSteps = 1000;

%Define the number of iterations that will be performed for timing.
iterations = 1;

%Run the application to generate the final heat map.
%This will only return the final heat map.
%Start the timer.
tic;

for i=1:iterations
	%Calculate
	updatedHeatArrayCPU = HeatTransferUpdateMatlab(heat_array, heatSpeed, timeSteps);
end

%Stop the timer.
CPUelapsedTime = toc;

%Print the amount of time that it took per iteration.
averageCPUTime = CPUelapsedTime / iterations;
fprintf('Average CPU time per iteration: %g seconds\n', averageCPUTime);
fprintf('    Average time per step: %g\n', averageCPUTime / timeSteps);

%Now, run the GPU implementation.
%Run a warmup pass.
updatedHeatArrayGPU = double(HeatTransferCUDA(single(heat_array), heatSpeed, timeSteps));

%Start the timer.
tic;

for i=1:iterations
	%Calculate
	updatedHeatArrayGPU = double(HeatTransferCUDA(single(heat_array), heatSpeed, timeSteps));
end

%Stop the timer.
GPUelapsedTime = toc;

%Print the amount of time that it took per iteration.
averageGPUTime = GPUelapsedTime / iterations;
fprintf('Average GPU time per iteration: %g seconds\n', averageGPUTime);
fprintf('    Average time per step: %g\n', averageGPUTime / timeSteps);

%Calculate the speedup.
speedup = averageCPUTime / averageGPUTime;
fprintf('Speedup = %g\n', speedup);

%Generate a plot for each of the implemenations for a visual comparison.
figure('Name', 'Initial Heat Map');
mesh(heat_array);
colorbar;
caxis([0 maxTemperature])
view(0,90);

figure('Name', 'CPU Output Heat Map');
mesh(updatedHeatArrayCPU);
colorbar;
caxis([0 maxTemperature])
view(0,90);

figure('Name', 'GPU Output Heat Map');
mesh(updatedHeatArrayGPU);
colorbar;
caxis([0 maxTemperature])
view(0,90);

%Compute the difference between the implementations.
delta = ( updatedHeatArrayGPU - updatedHeatArrayCPU ) .^ 2;
product = (updatedHeatArrayGPU .* updatedHeatArrayCPU);
deltaSum = sum(sum(delta));
productSum = sum(sum(product));
fprintf('Error between implementations: %g\n', sqrt(deltaSum / productSum));

%Finally, save the images that will be used to generate our little animation.
heat_array_save = heat_array;
for i=1:timeSteps
	heat_array_save = double(HeatTransferCUDA(single(heat_array_save), heatSpeed, 1));
	SaveIterationStep(heat_array_save, i, maxTemperature);
end

%Run the command to generate the MPEG video
dos('ffmpeg -r 60 -i image%04d.png -c:v libx264 -r 60 -pix_fmt yuv420p out.mp4');