% Define parameters
wavelength = 1550e-9;  % Wavelength in meters
beam_waist = 40e-6;   % Beam waist radius in meters
num_ports = 5;        % Number of output ports
desired_ratios = [1, 2, 3, 4, 5];  % Desired power ratios

% Create a grid for simulation
grid_size = 400;  % Adjust as needed
x = linspace(-beam_waist*5, beam_waist*5, grid_size);
y = linspace(-beam_waist*5, beam_waist*5, grid_size);
[X, Y] = meshgrid(x, y);

% Calculate the Gaussian beam profile
Gaussian_beam = exp(-(X.^2 + Y.^2) / (2 * (beam_waist^2)));

% Create a binary phase grating pattern
phase_grating = zeros(grid_size, grid_size);
for n = 1:num_ports
    phase_shift = 2 * pi * n * Gaussian_beam;  % Adjust phase as needed
    phase_grating = phase_grating + phase_shift;
end

% Apply the phase pattern to the incident beam
incident_beam = Gaussian_beam .* exp(1i * phase_grating);

% Normalize the incident beam
incident_beam = incident_beam / sqrt(sum(abs(incident_beam(:)).^2));

% Plot the incident beam
figure;
subplot(1, 2, 1);
imagesc(x, y, abs(incident_beam).^2);
title('Incident Beam Intensity');
xlabel('X (m)');
ylabel('Y (m)');
colormap('jet');
axis equal;
axis tight;

% Perform far-field diffraction to observe the output beams
output_beams = fftshift(fft2(incident_beam));
output_intensity = abs(output_beams).^2;

% Plot the output beams
subplot(1, 2, 2);
imagesc(x, y, output_intensity);
title('Output Beams Intensity');
xlabel('X (m)');
ylabel('Y (m)');
colormap('jet');
axis equal;
axis tight;

% Define angles for sectors (adjust as needed)
theta = linspace(0, 2*pi, num_ports+1);
theta_centers = (theta(1:end-1) + theta(2:end)) / 2;

% Initialize variables to store power in each output port
output_powers = zeros(1, num_ports);

% Calculate power in each output port
for i = 1:num_ports
    % Define the sector boundaries
    theta_start = theta(i);
    theta_end = theta(i+1);
    
    % Mask for the sector
    sector_mask = (atan2(Y, X) >= theta_start) & (atan2(Y, X) < theta_end);
    
    % Integrate the intensity over the sector
    output_powers(i) = sum(output_intensity(sector_mask));
end

% Normalize the output powers
total_power = sum(output_powers);
output_powers_normalized = output_powers / total_power;

% Display the power distribution in each output port
disp('Normalized Power Distribution in Output Ports:');
disp(output_powers_normalized);

% Calculate total splitting efficiency (TSE)
desired_total_power = sum(desired_ratios);
simulated_total_power = sum(output_powers_normalized);
TSE = simulated_total_power / desired_total_power;

% Calculate similarity using mean squared error (MSE)
MSE = mean((output_powers_normalized - desired_ratios).^2);
similarity = 1 / (1 + MSE);

% Display the results and metrics
disp(['Total Splitting Efficiency (TSE): ', num2str(TSE)]);
disp(['Similarity (SED): ', num2str(similarity)]);
% Define parameters
N = 5; % Number of output ports
P = [1, 2, 3, 4, 5]; % Power ratios of the sub-beams

% Initialize matrix to store amplitude and phase values
A = zeros(1, N);
phi = zeros(1, N);

% Set amplitude and phase values based on power ratios
for i = 1:N
    A(i) = sqrt(P(i));
    phi(i) = angle(exp(1i*2*pi*i/N));
end

% Generate metasurface phase pattern
[x, y] = meshgrid(linspace(-1, 1, 100));
phase_pattern = zeros(size(x));

for i = 1:N
    phase_pattern = phase_pattern + A(i)*exp(1i*phi(i) + 1i*2*pi*(i-1)/N*(x+y));
end

% Display the metasurface phase pattern
figure;
imagesc(phase_pattern);
axis equal;
colormap(jet);
colorbar;
title('Metasurface Phase Pattern');

% Apply the phase pattern to incident light beam
incident_beam = exp(1i*2*pi*(x+y));
output_beams = incident_beam.*exp(1i*phase_pattern);

% Display the output beams on the observation plane
figure;
for i = 1:N
    subplot(1, N, i);
    imagesc(abs(output_beams(:,:,i)).^2);
    axis equal;
    colormap(gray);
    title(sprintf('Output Beam %d', i));
end
% Define parameters
N = 4; % Number of output ports
P = [1, 3, 5, 2]; % Power ratios of the sub-beams

% Initialize matrix to store amplitude and phase values
A = zeros(1, N);
phi = zeros(1, N);

% Set amplitude and phase values based on power ratios
for i = 1:N
    A(i) = sqrt(P(i));
    phi(i) = angle(exp(1i*2*pi*i/N));
end

% Generate metasurface phase pattern
[x, y] = meshgrid(linspace(-1, 1, 100));
phase_pattern = zeros(size(x));

for i = 1:N
    phase_pattern = phase_pattern + A(i)*exp(1i*phi(i) + 1i*2*pi*(i-1)/N*(x+y));
end

% Display the metasurface phase pattern
figure;
imagesc(phase_pattern);
axis equal;
colormap(jet);
colorbar;
title('Metasurface Phase Pattern');

% Apply the phase pattern to incident light beam
incident_beam = exp(1i*2*pi*(x+y));
output_beams = incident_beam.*exp(1i*phase_pattern);

% Display the output beams on the observation plane
figure;
for i = 1:N
    subplot(1, N, i);
    imagesc(abs(output_beams(:,:,i)).^2);
    axis equal;
    colormap(gray);
    title(sprintf('Output Beam %d', i));
end
