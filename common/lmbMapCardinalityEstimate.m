function [nMap, mapIndices] = lmbMapCardinalityEstimate(r)
% LMBMAPCARDINALITYESTIMATE -- Determine approximate LMB MAP estimate
%   [nMap, mapIndices] = lmbMapCardinalityEstimate(r)
%
%   This function computes an approximate MAP estimate for the LMB filter.
%   文件导读：
%       LMB 输出阶段的 MAP 基数估计器。滤波器完成 posterior update 后，
%       用它决定当前时刻输出多少个 Bernoulli component，以及选择哪些
%       label 作为状态估计。
%
%   See also runLmbFilter.
%
%   Inputs
%       r - array. Each object's posterior existence probability.
%
%   Output
%       nMap - integer. The MAP estimate for the LMB
%           cardinality estimate.
%       mapIndices - array. The indices of the nMap greatest indices
%           of r

%% 1. 用 elementary symmetric function 计算 LMB cardinality distribution
r = r - 1e-6; % Does not work with unit existence probabilities
rho = prod(1 - r)*esf(r./(1-r));
%% 2. 选择 MAP cardinality，并返回存在概率最高的 nMap 个 component
[~, maxCardinalityIndex] = max(rho);
% The MAP estimate cannot be larger than the number of objects
nMap = min(maxCardinalityIndex - 1, length(r));
[~, sortedIndices] = sort(-r);
mapIndices = sortedIndices(1:nMap);
end
