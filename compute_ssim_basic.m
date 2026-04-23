function val = compute_ssim_basic(ref, test)
%% COMPUTE_SSIM_BASIC   -   Compute SSIM between two image pairs, ref and test
%  The implementation originates from the standard SSIM formula with an 11x11 Gaussian
%  window of standard deviation 1.5.

    ref = clip01(ref);
    test = clip01(test);

    if ~isequal(size(ref), size(test))
        error('Reference and test images must have the same size.');
    end

    ref = double(ref);
    test = double(test);

    winSize = 11;
    sigma = 1.5;
    kernel = gaussian_kernel(winSize, sigma);

    mu1 = conv2(ref, kernel, 'same');
    mu2 = conv2(test, kernel, 'same');

    mu1Sq = mu1 .^ 2;
    mu2Sq = mu2 .^ 2;
    mu1mu2 = mu1 .* mu2;

    sigma1Sq = conv2(ref .^ 2, kernel, 'same') - mu1Sq;
    sigma2Sq = conv2(test .^ 2, kernel, 'same') - mu2Sq;
    sigma12 = conv2(ref .* test, kernel, 'same') - mu1mu2;

    sigma1Sq = max(sigma1Sq, 0);
    sigma2Sq = max(sigma2Sq, 0);

    L = 1;
    C1 = (0.01 * L) ^ 2;
    C2 = (0.03 * L) ^ 2;

    numerator = (2 * mu1mu2 + C1) .* (2 * sigma12 + C2);
    denominator = (mu1Sq + mu2Sq + C1) .* (sigma1Sq + sigma2Sq + C2);

    ssimMap = numerator ./ max(denominator, eps);

    border = floor(winSize / 2);
    if size(ssimMap, 1) > 2 * border && size(ssimMap, 2) > 2 * border
        ssimMap = ssimMap(border + 1:end - border, border + 1:end - border);
    end

    val = mean(ssimMap(:));
end

function kernel = gaussian_kernel(winSize, sigma)
%% GAUSSIAN_KERNEL -    Generate a normalized 2D Gaussian kernel.

    radius = (winSize - 1) / 2;
    [x, y] = meshgrid(-radius:radius, -radius:radius);
    kernel = exp(-(x .^ 2 + y .^ 2) / (2 * sigma ^ 2));
    kernel = kernel / sum(kernel(:));
end
