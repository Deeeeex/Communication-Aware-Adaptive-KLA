function model = generateAdvancedModel(varargin)
%GENERATEADVANCEDMODEL - Enhanced model generation with comprehensive configuration
%   model = generateAdvancedModel() creates a model with default parameters
%
%   model = generateAdvancedModel(Name, Value) allows specification of model
%   parameters using name-value pairs.
%
%   Mathematical Framework:
%   The model implements the Labeled Multi-Bernoulli (LMB) filter framework
%   where the multi-object state is represented as:
%
%   $$X = \{(x^{(1)}, \ell^{(1)}), \ldots, (x^{(N)}, \ell^{(N)})\}$$
%
%   where $x^{(i)}$ is the kinematic state and $\ell^{(i)}$ is the unique label
%   of object $i$.
%
%   The LMB density is parameterized as:
%   $$\pi(X) = \Delta(X) \prod_{i=1}^{N} \frac{r^{(\ell^{(i)})}}{1-r^{(\ell^{(i)})}} p^{(\ell^{(i)})}(x^{(i)})$$
%
%   where:
%   - $\Delta(X)$ ensures distinct labels
%   - $r^{(\ell)}$ is existence probability for label $\ell$
%   - $p^{(\ell)}(x)$ is spatial density for label $\ell$
%
%   Name-Value Parameters:
%       'ClutterRate'           - Expected clutter per time step (default: 10)
%       'DetectionProbability'  - Target detection probability (default: 0.95)
%       'DataAssociation'       - Algorithm: 'LBP', 'Gibbs', 'Murty' (default: 'LBP')
%       'ScenarioType'          - 'Fixed' or 'Random' (default: 'Fixed')
%       'NumberOfTargets'       - For random scenarios (default: 10)
%       'SurvivalProbability'   - Target survival probability (default: 0.99)
%       'ProcessNoise'          - Process noise variance (default: 1.0)
%       'MeasurementNoise'      - Measurement noise variance (default: 1.0)
%       'StateTransition'       - State transition model (default: 'ConstantVelocity')
%       'MeasurementModel'      - Measurement model (default: 'Position')
%       'BirthIntensity'        - Birth intensity (default: 0.1)
%       'ExistenceThreshold'    - Track pruning threshold (default: 0.1)
%       'LBPTolerance'          - LBP convergence tolerance (default: 1e-6)
%       'LBPMaxIterations'      - Maximum LBP iterations (default: 1000)
%       'GibbsSamples'          - Number of Gibbs samples (default: 5000)
%       'MurtyAssignments'      - Number of Murty assignments (default: 100)
%
%   Output:
%       model - Struct containing all model parameters and matrices
%
%   State Space Model:
%   The dynamics follow a linear Gaussian model:
%   $$x_{k+1} = F x_k + G w_k$$
%   $$z_k = H x_k + v_k$$
%
%   where:
%   - $F$ is the state transition matrix
%   - $G$ is the process noise gain matrix
%   - $H$ is the measurement matrix
%   - $w_k \sim \mathcal{N}(0, Q)$ is process noise
%   - $v_k \sim \mathcal{N}(0, R)$ is measurement noise
%
%   Examples:
%       % Basic model with default parameters
%       model = generateAdvancedModel();
%
%       % High-clutter scenario
%       model = generateAdvancedModel('ClutterRate', 20, ...
%                                   'DetectionProbability', 0.8);
%
%       % Random scenario with many targets
%       model = generateAdvancedModel('ScenarioType', 'Random', ...
%                                   'NumberOfTargets', 25, ...
%                                   'DataAssociation', 'Gibbs');
%
%   See also generateMultisensorModel, generateGroundTruth

%% Input parsing
p = inputParser;

% Core parameters
addParameter(p, 'ClutterRate', 10, @(x) isnumeric(x) && x > 0);
addParameter(p, 'DetectionProbability', 0.95, @(x) isnumeric(x) && x > 0 && x <= 1);
addParameter(p, 'DataAssociation', 'LBP', @(x) ischar(x) && ismember(x, {'LBP', 'Gibbs', 'Murty', 'LBPFixed'}));
addParameter(p, 'ScenarioType', 'Fixed', @(x) ischar(x) && ismember(x, {'Fixed', 'Random'}));
addParameter(p, 'NumberOfTargets', 10, @(x) isnumeric(x) && x > 0);

% Physical parameters
addParameter(p, 'SurvivalProbability', 0.99, @(x) isnumeric(x) && x > 0 && x <= 1);
addParameter(p, 'ProcessNoise', 1.0, @(x) isnumeric(x) && x > 0);
addParameter(p, 'MeasurementNoise', 1.0, @(x) isnumeric(x) && x > 0);
addParameter(p, 'StateTransition', 'ConstantVelocity', @(x) ischar(x));
addParameter(p, 'MeasurementModel', 'Position', @(x) ischar(x));
addParameter(p, 'BirthIntensity', 0.1, @(x) isnumeric(x) && x > 0);

% Algorithm parameters
addParameter(p, 'ExistenceThreshold', 0.1, @(x) isnumeric(x) && x > 0 && x < 1);
addParameter(p, 'LBPTolerance', 1e-6, @(x) isnumeric(x) && x > 0);
addParameter(p, 'LBPMaxIterations', 1000, @(x) isnumeric(x) && x > 0);
addParameter(p, 'GibbsSamples', 5000, @(x) isnumeric(x) && x > 0);
addParameter(p, 'MurtyAssignments', 100, @(x) isnumeric(x) && x > 0);

parse(p, varargin{:});
params = p.Results;

%% Initialize model structure
model = struct();

%% Basic simulation parameters
model.simulationLength = 100;
model.dT = 1; % Time step
model.xDimension = 4; % [x, y, vx, vy]
model.zDimension = 2; % [x, y]

%% Store configuration parameters
model.clutterRate = params.ClutterRate;
model.detectionProbability = params.DetectionProbability;
model.dataAssociationMethod = params.DataAssociation;
model.scenarioType = params.ScenarioType;
model.numberOfTargets = params.NumberOfTargets;
model.survivalProbability = params.SurvivalProbability;
model.existenceThreshold = params.ExistenceThreshold;

%% Generate state transition matrix F for constant velocity model
% State vector: [x, y, vx, vy]'
% Discrete-time constant velocity model:
% x(k+1) = x(k) + T*vx(k)
% y(k+1) = y(k) + T*vy(k)
% vx(k+1) = vx(k)
% vy(k+1) = vy(k)
switch params.StateTransition
    case 'ConstantVelocity'
        model.F = [1, 0, model.dT, 0;
                   0, 1, 0, model.dT;
                   0, 0, 1, 0;
                   0, 0, 0, 1];
    case 'ConstantAcceleration'
        model.F = [1, 0, model.dT, 0, 0.5*model.dT^2, 0;
                   0, 1, 0, model.dT, 0, 0.5*model.dT^2;
                   0, 0, 1, 0, model.dT, 0;
                   0, 0, 0, 1, 0, model.dT;
                   0, 0, 0, 0, 1, 0;
                   0, 0, 0, 0, 0, 1];
        model.xDimension = 6; % [x, y, vx, vy, ax, ay]
    otherwise
        error('Unknown state transition model: %s', params.StateTransition);
end

%% Process noise covariance matrix Q
% Continuous-time white noise acceleration model
switch params.StateTransition
    case 'ConstantVelocity'
        % Process noise gain matrix
        G = [0.5*model.dT^2, 0;
             0, 0.5*model.dT^2;
             model.dT, 0;
             0, model.dT];
        
        % Process noise covariance
        Q_continuous = params.ProcessNoise * eye(2);
        model.Q = G * Q_continuous * G';
        
    case 'ConstantAcceleration'
        % For constant acceleration model
        G = [model.dT^3/6, 0;
             0, model.dT^3/6;
             model.dT^2/2, 0;
             0, model.dT^2/2;
             model.dT, 0;
             0, model.dT];
        
        Q_continuous = params.ProcessNoise * eye(2);
        model.Q = G * Q_continuous * G';
end

%% Measurement model matrix H
switch params.MeasurementModel
    case 'Position'
        model.H = [1, 0, 0, 0;
                   0, 1, 0, 0];
        if model.xDimension == 6
            model.H = [1, 0, 0, 0, 0, 0;
                       0, 1, 0, 0, 0, 0];
        end
    case 'PositionVelocity'
        model.H = eye(model.xDimension);
        model.zDimension = model.xDimension;
    otherwise
        error('Unknown measurement model: %s', params.MeasurementModel);
end

%% Measurement noise covariance matrix R
model.R = params.MeasurementNoise * eye(model.zDimension);

%% For compatibility with existing code
model.C = model.H; % Measurement matrix
model.sigmaQ = sqrt(params.ProcessNoise); % Process noise standard deviation
model.sigmaR = sqrt(params.MeasurementNoise); % Measurement noise standard deviation

%% Clutter parameters
model.clutterPerUnitVolume = model.clutterRate / (100 * 100); % Assume 100x100 surveillance area
model.surveillanceRegion = [-50, 50; -50, 50]; % [x_min, x_max; y_min, y_max]

%% Birth model parameters
model.birthIntensity = params.BirthIntensity;
model.numberOfBirthLocations = 4; % Default for fixed scenario

if strcmp(params.ScenarioType, 'Fixed')
    % Fixed birth locations (corners of surveillance region)
    model.birthLocations = [
        -40, -40;  % Bottom-left
         40, -40;  % Bottom-right
         40,  40;  % Top-right
        -40,  40   % Top-left
    ];
else
    % Random birth locations
    model.numberOfBirthLocations = params.NumberOfTargets;
    model.birthLocations = generateRandomBirthLocations(params.NumberOfTargets, model.surveillanceRegion);
end

%% Initialize birth components
model.birthTrajectory = struct();
for i = 1:model.numberOfBirthLocations
    model.birthTrajectory(i).birthTime = 1;
    model.birthTrajectory(i).birthLocation = i;
    model.birthTrajectory(i).trajectoryLength = 0;
    model.birthTrajectory(i).r = model.birthIntensity;
    model.birthTrajectory(i).w = 1; % Single Gaussian component
    model.birthTrajectory(i).numberOfGmComponents = 1;
    
    % Initial state: [x, y, vx, vy]
    initial_position = model.birthLocations(i, :)';
    initial_velocity = [0; 0]; % Start at rest
    model.birthTrajectory(i).mu = {[initial_position; initial_velocity]};
    
    % Initial covariance
    initial_pos_var = 10; % Position uncertainty
    initial_vel_var = 5;  % Velocity uncertainty
    model.birthTrajectory(i).Sigma = {diag([initial_pos_var, initial_pos_var, initial_vel_var, initial_vel_var])};
end

%% Algorithm-specific parameters
switch params.DataAssociation
    case 'LBP'
        model.lbpConvergenceTolerance = params.LBPTolerance;
        model.maximumNumberOfLbpIterations = params.LBPMaxIterations;
    case 'LBPFixed'
        model.maximumNumberOfLbpIterations = params.LBPMaxIterations;
    case 'Gibbs'
        model.numberOfSamples = params.GibbsSamples;
        model.gibbsBurnIn = round(params.GibbsSamples * 0.2); % 20% burn-in
    case 'Murty'
        model.numberOfAssignments = params.MurtyAssignments;
end

%% Track management parameters
model.minimumTrajectoryLength = 5;
model.maximumNumberOfObjects = 50;
model.weightThreshold = 1e-4;
model.maximumComponents = 10;

%% Object initialization for empty scenario
model.object = model.birthTrajectory([]); % Empty initial set

%% LMBM-specific parameters (for hypothesis management)
model.hypotheses = struct();
model.hypotheses.r = [];
model.hypotheses.w = 1;
model.hypotheses.numberOfObjects = 0;

%% Performance monitoring
model.enableProfiling = false;
model.saveIntermediateResults = false;

%% Validation
validateModel(model);

end

%% Helper function to generate random birth locations
function locations = generateRandomBirthLocations(numLocations, region)
    x_range = region(1, :);
    y_range = region(2, :);
    
    locations = zeros(numLocations, 2);
    locations(:, 1) = x_range(1) + (x_range(2) - x_range(1)) * rand(numLocations, 1);
    locations(:, 2) = y_range(1) + (y_range(2) - y_range(1)) * rand(numLocations, 1);
end

%% Model validation function
function validateModel(model)
    % Validate state transition matrix
    if size(model.F, 1) ~= model.xDimension || size(model.F, 2) ~= model.xDimension
        error('State transition matrix F has incorrect dimensions');
    end
    
    % Validate measurement matrix
    if size(model.H, 1) ~= model.zDimension || size(model.H, 2) ~= model.xDimension
        error('Measurement matrix H has incorrect dimensions');
    end
    
    % Validate noise covariances
    if any(eig(model.Q) <= 0)
        warning('Process noise covariance Q is not positive definite');
    end
    
    if any(eig(model.R) <= 0)
        warning('Measurement noise covariance R is not positive definite');
    end
    
    % Validate probabilities
    if model.detectionProbability <= 0 || model.detectionProbability > 1
        error('Detection probability must be in (0, 1]');
    end
    
    if model.survivalProbability <= 0 || model.survivalProbability > 1
        error('Survival probability must be in (0, 1]');
    end
    
    % Validate birth model
    if isempty(model.birthTrajectory)
        warning('No birth components defined');
    end
    
    fprintf('Model validation completed successfully.\n');
end
