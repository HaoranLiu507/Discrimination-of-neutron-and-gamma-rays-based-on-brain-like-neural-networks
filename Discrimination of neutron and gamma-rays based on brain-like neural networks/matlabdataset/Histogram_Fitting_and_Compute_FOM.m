function [y_counts,miu, sigma, FOM] = Histogram_Fitting_and_Compute_FOM(Pulse_shape_discrimination_factor)
    R = Pulse_shape_discrimination_factor;

    % Normalization Pulse_shape_discrimination_factor and magnify Pulse_shape_discrimination_factor
    R = mapminmax(R, 0, 1);
    R = R * 200;

    % Histogram of Pulse_shape_discrimination_factor
    [counts, edges] = histcounts(R, 'Normalization', 'count'); 
    bin_centers = (edges(1:end-1) + edges(2:end)) / 2; 

    % Double Gaussian fitting
    num_components = 2; 
    options = statset('MaxIter', 1000);
    disp(size(R))
    gmModel = fitgmdist(R, num_components, 'Options', options);

    % Generate x values for plotting
    x = linspace(min(R), max(R), 1000);

    % Get probability density from the Gaussian model
    y_pdf = pdf(gmModel, x');

    % Calculate corresponding counts using the width of the bins
    bin_width = edges(2) - edges(1);
    y_counts = y_pdf * sum(counts) * bin_width; % Convert PDF to counts

    % Plotting the histogram and fitting results
    figure;
    hold on;
    bar(bin_centers, counts, 'FaceAlpha', 0.5, 'EdgeColor', 'none'); % Plot histogram
    plot(x, y_counts, 'LineWidth', 2, 'Color', 'r'); % Plot Gaussian fit
    title('Double Gaussian Fitting with Histogram Counts');
    xlabel('Pulse Shape Discrimination Factor');
    ylabel('Counts');
    legend('Histogram', 'Gaussian Fit');

    % Display mean and standard deviation
    miu = gmModel.mu;
    sigma = sqrt(gmModel.Sigma);

    % Compute FOM
    FOM = abs((miu(2) - miu(1)) / (1.667 * (sigma(2) + sigma(1))));
    str_FOM = sprintf('FOM = %.2f', FOM);
    dim = [.70 .70 .2 .1];
    annotation('textbox', dim, 'String', str_FOM, 'FontSize', 12, 'FitBoxToText', 'on');
    hold off;
end