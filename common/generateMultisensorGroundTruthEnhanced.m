function [groundTruth, measurements, groundTruthRfs, sensorTrajectories] = generateMultisensorGroundTruthEnhanced(model, varargin)
% GENERATEMULTISENSORGROUNDTRUTHENHANCED - Wrapper for extended motion support
%   [groundTruth, measurements, groundTruthRfs, sensorTrajectories] =
%   generateMultisensorGroundTruthEnhanced(model, varargin)
%
%   This wrapper delegates to generateMultisensorGroundTruth, which now
%   supports CV, CT, and Formation sensor motion as well as formation targets.
%   文件导读：
%       旧 enhanced-motion 脚本的兼容入口。当前真实实现已经集中在
%       generateMultisensorGroundTruth；阅读核心逻辑时优先看那个文件。
%
%   See also generateMultisensorGroundTruth

[groundTruth, measurements, groundTruthRfs, sensorTrajectories] = generateMultisensorGroundTruth(model, varargin{:});
end
