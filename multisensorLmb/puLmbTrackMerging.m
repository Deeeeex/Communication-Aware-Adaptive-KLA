function objects = puLmbTrackMerging(measurementUpdatedDistributions, objects, model)
% PULMBTRACKMERGING -- Merge the objects' measurement-updated distributions together
%   objects = puLmbTrackMerging(measurementUpdatedDistributions, objects, model)
%
%   Merge the objects' measurement-updated distributions together while
%   assuming independent sensors. Very crude merging algorithm, it might be
%   better use expectation propagation (if possible).
%   文件导读：
%       PU 融合路径。它假设传感器条件独立，直接合并各传感器的
%       measurement-updated posterior，并补偿重复使用的 prior。注意：
%       当前动态 GA/AA 权重不作用于 PU，本文件主要用于融合规则对照。
%
%   See also generateMultisensorModel, loopyBeliefPropagation, lmbGibbsSampling, lmbMurtysAlgorithm
%
%   Inputs
%       measurementUpdatedDistributions - (1, numberOfSensors) cell array.
%           Each is an object struct produced by computePosteriorLmbSpatialDistributions.
%       objects - struct. A struct containing the prior LMB's Bernoulli
%           components. This struct is produced by lmbPredictionStep.
%       model - struct. A struct with the fields declared in generateMultisensorModel.
%
%   Output
%       objects - struct. A struct containing the posterior LMB's Bernoulli
%           components.


%% 1. 逐 Bernoulli component 合并各传感器的 measurement-updated posterior
for i = 1:numel(objects)
    %% 1.1 计算跨传感器 Gaussian mixture 的笛卡尔积大小
    numberOfGmComponents = zeros(1, model.numberOfSensors);
    for s = 1:model.numberOfSensors
        numberOfGmComponents(s) = measurementUpdatedDistributions{s}(i).numberOfGmComponents;
    end
    %% 1.2 将 prior 转成 canonical form，用于补偿多次使用 prior
    KPrior = inv(objects(i).Sigma{1});
    hPrior = KPrior * objects(i).mu{1};
    gPrior = -0.5 * objects(i).mu{1}' * KPrior * objects(i).mu{1} - 0.5 * log(det(2*pi*objects(i).Sigma{1}));
    %% 1.3 预分配融合后 mixture；初值包含 prior 补偿项
    numberOfPosteriorGmComponents = prod(numberOfGmComponents);
    K = repmat({ (1 - model.numberOfSensors) * KPrior  }, 1, numberOfPosteriorGmComponents);
    h = repmat({ (1 - model.numberOfSensors) * hPrior  }, 1, numberOfPosteriorGmComponents);
    g = repmat((1 - model.numberOfSensors) * gPrior, 1, numberOfPosteriorGmComponents);
    %%
    for s = 1:model.numberOfSensors
        % Setup intermediate mxiture
        currentMixtureSize = prod(numberOfGmComponents(1:(s-1)));
        KI = K;
        hI = h;
        gI = g;
        ell = 0;
        for j = 1:numberOfGmComponents(s)
            % Convert to canonical from
            % Local names
            w = measurementUpdatedDistributions{s}(i).w(j);
            mu = measurementUpdatedDistributions{s}(i).mu{j};
            Sigma = measurementUpdatedDistributions{s}(i).Sigma{j};
            % Convert to canonical from 
            KC = inv(Sigma);
            hC = KC * mu;
            gC = -0.5 * mu' * (KC * mu) - 0.5 * log(det(2 * pi * Sigma)) + log(w);
            for k = 1:currentMixtureSize
                ell = ell + 1;
                K{ell} = KI{k} + KC;
                h{ell} = hI{k} + hC;
                g(ell) = gI(k) + gC;
            end
        end
    end
    %% 1.4 转回 covariance form 并归一化 mixture 权重
    for j = 1:numberOfPosteriorGmComponents
        T = K{j};
        % Reuse variable names
        K{j} = inv(K{j});
        h{j} = K{j} * h{j};
        g(j) = g(j) + 0.5 * h{j}' * T * h{j} + 0.5 * log(det(2 * pi * K{j}));
    end
    eta = sum(exp(g));
    w = exp(g - max(g));
    w = w ./ sum(w);
    %% 1.5 合并 Bernoulli existence probability
    numerator = eta * (objects(i).r)^(1 - model.numberOfSensors);
    partialDenominator = (1 - objects(i).r)^(1 - model.numberOfSensors);
    for s = 1:model.numberOfSensors
        rS = measurementUpdatedDistributions{s}(i).r;
        numerator = numerator * rS;
        partialDenominator = partialDenominator *  (1 - rS);
    end
    %% Update Bernoulli component
    % Sort components according to weight, and discard insignificants components
    [~, maxIndex] = max(w);
    % Update Bernoulli component
    objects(i).r = numerator / (numerator + partialDenominator);
    objects(i).numberOfGmComponents = 1;
    objects(i).w = 1;
    objects(i).mu = h(maxIndex);
    objects(i).Sigma = K(maxIndex);
end
end
