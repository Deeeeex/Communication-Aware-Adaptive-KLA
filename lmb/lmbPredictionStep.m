function objects = lmbPredictionStep(objects, model, t)
% LMBPREDICTIONSTEP -- Complete the LMB filter's prediction step.
%   objects = lmbPredictionStep(objects, model, t)
%
%   Computes predicted prior for the current time-step using the 
%   Chapman-Kolmogorov equation, assuming an LMB prior and the standard 
%   multi-object motion model. 
%   文件导读：
%       LMB 的共享预测步骤，单传感器和多传感器 LMB 主循环都会调用。
%       它先把已有 Bernoulli component 按线性高斯运动模型传播，再把
%       当前时刻的新生 birth components 追加到对象集合中。
%
%   See also runLmbFilter, generateModel
%
%   Inputs
%       objects - struct. A struct containing the posterior LMB's Bernoulli components.
%       model - struct. A struct with the fields declared in generateModel.
%       t - integer. An integer representing the simulation's current
%           time-step
%
%   Output
%       objects - struct. A struct containing the prior LMB's Bernoulli components.

%% 1. 传播已有 Bernoulli component：存在概率乘生存概率，Gaussian 做 Kalman prediction
numberOfObjects = numel(objects);
for i = 1:numberOfObjects
    objects(i).r = model.survivalProbability * objects(i).r;
    for j = 1:objects(i).numberOfGmComponents
        objects(i).mu{j} = model.A * objects(i).mu{j} + model.u;
        objects(i).Sigma{j} = model.A * objects(i).Sigma{j} * model.A' + model.R;
    end
end
%% 2. 追加 birth components：每个 birth location 在当前时刻生成一个候选 Bernoulli
newNumberOfObjects = numberOfObjects + model.numberOfBirthLocations;
objects(numberOfObjects+1:newNumberOfObjects) = model.birthParameters;
[objects(numberOfObjects+1:newNumberOfObjects).birthTime] = deal(t);
end
