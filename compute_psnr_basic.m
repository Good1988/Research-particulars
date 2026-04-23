function val = compute_psnr_basic(ref, test)
%% COMPUTE_PSNR_BASIC - Compute PSNR between two image pairs, ref and test, assumed to have intensities within [0, 1].

    ref = clip01(ref);
    test = clip01(test);

    if ~isequal(size(ref), size(test))
        error('Reference and test images must have the same size.');
    end

    mse = mean((ref(:) - test(:)) .^ 2);

    if mse <= eps
        val = Inf;
    else
        val = 10 * log10(1 / mse);
    end
end
