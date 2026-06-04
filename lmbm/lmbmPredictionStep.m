function hypothesis = lmbmPredictionStep(hypothesis, model, t)
% LMBMPREDICTIONSTEP -- Complete the LMBM filter's prediction step.
%   objects = lmbmPredictionStep(objects, model, t)
%
%   Computes predicted prior for the current time-step using the 
%   Chapman-Kolmogorov equation, assuming an LMBM prior and the standard 
%   multi-object motion model. 
%   文件导读：
%       LMBM 版本的 prediction step。它对一个 global hypothesis 内的每个
%       Bernoulli component 做运动预测，并在 measurement-update branching
%       之前追加当前时刻的新生 components。
%
%   See also runLmbmFilter, generateModel
%
%   Inputs
%       hypothesis - struct. A struct containing a posterior LMBM hypothesis' Bernoulli components.
%       model - struct. A struct with the fields declared in generateModel.
%       t - integer. An integer representing the simulation's current
%           time-step
%
%   Output
%       hypothesis - struct. A struct containing the prior LMBM hyopthesis' Bernoulli components.

%% 1. 预测已有 Bernoulli component
numberOfObjects = numel(hypothesis.r);
for i = 1:numberOfObjects
    hypothesis.r(i) = model.survivalProbability * hypothesis.r(i);
    hypothesis.mu{i} = model.A * hypothesis.mu{i} + model.u;
    hypothesis.Sigma{i} = model.A * hypothesis.Sigma{i} * model.A' + model.R;
end
%% 2. 追加当前时刻 birth components
stride = (numberOfObjects + 1):(numberOfObjects + model.numberOfBirthLocations);
hypothesis.birthLocation(stride) = model.birthLocationLabels;
hypothesis.birthTime(stride) = t;
hypothesis.r(stride, :) = model.rBLmbm;
hypothesis.mu(stride) = model.muB;
hypothesis.Sigma(stride) = model.SigmaB;
end
