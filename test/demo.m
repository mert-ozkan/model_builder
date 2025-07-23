%% ExponentialPowerLaw and NLLFitter demo

% Initiate model
mdl = ExponentialPowerLaw(verbose=false);

% Simulate data
x_data = .1:.001:120; % frequencies
p = [6, 2, 80]; % intercept, exponent, knee
sigma = .1; % noise

y_sim = mdl.simulate(p, sigma, x_data);
y_actual = mdl.predict(p, x_data);
y_wo_knee = p(1) .* (x_data .^ -p(2));
% Plot
figure;
plot(x_data, 10.^y_sim);
hold on;
plot(x_data, 10.^y_actual, 'LineWidth', 2);
plot(x_data, y_wo_knee, ':', 'LineWidth', 2);
xline(p(3));
set(gca,xscale='log', yscale='log');

%% MLE by NLLFitter
% Feed data to model
mdl.X = x_data;
mdl.Y = y_sim;

% Construct NLLFitter
fitter = NLLFitter(mdl);
p0 = fitter.estimate();


% Create a function handle for the objective function. fmincon requires
% a function that takes only one argument (the parameter vector 'p').
objective = @(p) fitter.objective_function(p);
hessian_func = @(p, lambda) fitter.compute_hessian(p);

% Set up the optimization options for fmincon.
% We tell the optimizer that our objective function will return the
% gradient and the Hessian, which makes the optimization much more efficient.
options = optimoptions('fmincon', ...
    'Display', 'iter', ...
    'Algorithm', 'interior-point', ... % This algorithm can use the Hessian
    'SpecifyObjectiveGradient', true, ...
    'HessianFcn', hessian_func);

% Define lower bounds (parameters must be positive)
lb = zeros(size(p0)); 

fprintf('--- Running fmincon for Maximum Likelihood Estimation ---\n');

% Run the optimization.
% We provide empty arrays [] for linear and nonlinear constraints as we don't have any.
[p_mle, nll_final] = fmincon(objective, p0, [], [], [], [], lb, [], [], options);


% --- 4. Display Results ---
fprintf('\n--- Results ---\n');
fprintf('True Parameters:      [%.2f, %.2f, %.2f, sigma=%.2f]\n', p, sigma);
fprintf('Final MLE Parameters: [%.2f, %.2f, %.2f, sigma=%.2f]\n', p_mle);
fprintf('Final Negative Log-Likelihood: %.4f\n', nll_final);


%% Sum of Gaussians
n_peaks = 3;
peaks = SumOfGaussians(n_peaks=n_peaks, verbose=false);

b = -.1;

P = [2, 20, 1;...
    1.5, 8, 1.8;...
    .7, 40, .5]; %amp, center, sd
sigma = .1;
x_data = linspace(.1, 120, 10^3);
y_sim = peaks.simulate(P, b, sigma,x_data);
y_actual = peaks.predict(P, b, x_data);

%% MLE by NLLFitter
% Feed data to model
peaks.X = x_data;
peaks.Y = y_sim;

% Construct NLLFitter
fitter = NLLFitter(peaks);
p0 = fitter.estimate(min_peak_width = .2);

% Create a function handle for the objective function. fmincon requires
% a function that takes only one argument (the parameter vector 'p').
objective = @(p) fitter.objective_function(p);
hessian_func = @(p, lambda) fitter.compute_hessian(p);

% Set up the optimization options for fmincon.
% We tell the optimizer that our objective function will return the
% gradient and the Hessian, which makes the optimization much more efficient.
options = optimoptions('fmincon', ...
    'Display', 'iter', ...
    'Algorithm', 'interior-point', ... % This algorithm can use the Hessian
    'SpecifyObjectiveGradient', true, ...
    'HessianFcn', hessian_func);
% Define lower bounds (parameters must be positive)

peak_lb = repmat([.1, .1, .1], [n_peaks,1]);
peak_ub = repmat([10, max(x_data), 5], [n_peaks,1]);
b_lb = -.1;
b_ub = .1;
s_lb = 0;
s_ub = 10;


lb = [peak_lb(:)', b_lb, s_lb];
ub = [peak_ub(:)', b_ub, s_ub];

fprintf('--- Running fmincon for Maximum Likelihood Estimation ---\n');

% Run the optimization.
% We provide empty arrays [] for linear and nonlinear constraints as we don't have any.
[p_mle, nll_final] = fmincon(objective, p0, [], [], [], [], lb, ub, [], options);

y_pred = peaks.predict(p_mle, x_data);

figure;
plot(x_data, y_sim);
hold on
plot(x_data, y_actual);
plot(x_data, y_pred);