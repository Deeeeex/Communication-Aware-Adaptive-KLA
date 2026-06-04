function r = poissrnd(lambda, varargin)
% POISSRND - Minimal Poisson sampler for Octave environments without statistics pkg.

if nargin < 1
    error('poissrnd requires at least one input.');
end

if any(lambda(:) < 0) || ~isreal(lambda)
    error('lambda must be real and nonnegative.');
end

if nargin == 1
    outSize = size(lambda);
    if isscalar(lambda)
        lambda = repmat(lambda, outSize);
    end
else
    if nargin == 2 && numel(varargin{1}) > 1
        outSize = varargin{1};
    else
        outSize = [varargin{:}];
    end
    if isempty(outSize)
        outSize = size(lambda);
    end
    if isscalar(lambda)
        lambda = repmat(lambda, outSize);
    elseif ~isequal(size(lambda), outSize)
        error('Non-scalar lambda must match the requested output size.');
    end
end

r = zeros(outSize);
for idx = 1:numel(r)
    lam = lambda(idx);
    if lam == 0
        r(idx) = 0;
        continue;
    end
    threshold = exp(-lam);
    product = 1.0;
    count = -1;
    while product > threshold
        count = count + 1;
        product = product * rand();
    end
    r(idx) = count;
end
end
