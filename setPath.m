% SETPATH -- Set up all the paths necessary to use the toolbox
%
% 文件导读：
%   MATLAB 会话的路径初始化入口。运行 demo 或手动调用滤波器前，先在
%   仓库根目录执行一次本脚本。这里只添加源码目录，不把 RUN 报告、
%   论文模板和输出产物加入 path，避免 MATLAB 误扫生成物。
rootDir = fileparts(mfilename('fullpath'));
addpath(fullfile(rootDir, 'common'));
addpath(fullfile(rootDir, 'lmb'));
addpath(fullfile(rootDir, 'lmbm'));
addpath(fullfile(rootDir, 'marginalEvalulations'));
addpath(fullfile(rootDir, 'trials'));
addpath(fullfile(rootDir, 'multisensorLmb'));
addpath(fullfile(rootDir, 'multisensorLmbm'));
