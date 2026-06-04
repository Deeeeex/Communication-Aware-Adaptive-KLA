function [cardinalityEstimate, extractionIndices] = lmbmStateExtraction(hypotheses, useEapOnLmbm)
% LMBMSTATEEXTRACTION -- Heuristically determine the number of objects
% present, and their respective parameters' indices.
%    [cardinalityEstimate, extractionIndices] = lmbmStateExtraction(hypotheses, useEapOnLmbm)
%
%   This function computes an approximate cardinality estimate for the LMBM filter.
%   文件导读：
%       LMBM 的状态抽取助手。它根据 posterior hypotheses 的存在概率汇总，
%       决定输出基数和 component 索引。
%
%   See also runLmbmFilter
%
%   Inputs
%       hypotheses - struct. A struct containing posterior LMBM
%           hypotheses, but with unnormalised hypothesis weights.
%       useEapOnLmbm - bool. A 'true' if we want to use an approximate EAP for 
%           state extraction, a 'false' we want to use an heuristic MAP.
%
%   Output
%       cardinalityEstimate - integer. An estimate of the number of objects present.
%       extractionIndices - array. An array of indices of the objects' with
%           the greatest existence probabilities.

rTotal = sum([hypotheses.w] .* [hypotheses.r], 2);
if (useEapOnLmbm)
    %% 1. EAP 风格抽取：用期望基数作为输出数量
    cardinalityEstimate = floor(sum(rTotal));
    [~,  extractionIndices] = maxk(hypotheses(1).r, cardinalityEstimate);
else
    %% 2. MAP 风格抽取：复用 LMB 的 MAP cardinality helper
    [cardinalityEstimate, extractionIndices] = lmbMapCardinalityEstimate([rTotal]);
end

end
