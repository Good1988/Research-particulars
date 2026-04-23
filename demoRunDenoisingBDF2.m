function demoRunDenoisingBDF2()
%% DEMODENOISINGBDF2    -   Demo script for running the codes using adaptive joint lambda-K selection.
%
% What the script does:
%   (1) Loads the demo noisy and clean images
%   (2) Searches jointly for lambda and K
%   (3) Denoises the noisy image using the selected pair (lambda, K)
%   (4) Saves the result and the parameter-search history.

    clc;
    close all;

    thisFile = mfilename('fullpath');
    thisDir = fileparts(thisFile);

    inputFile = fullfile(thisDir, 'data', '03Noise.png'); % Noisy image
    cleanFile = fullfile(thisDir, 'data', '03.png'); % Clean image
    outputDir = fullfile(thisDir, 'results');
    outputImageFile = fullfile(outputDir, 'denoisingBDF2_Optimal_lambda_K.png');
    historyFile = fullfile(outputDir, 'adaptive_lambda_K_history.csv');
    summaryFile = fullfile(outputDir, 'adaptive_lambda_K_summary.txt');

    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    noisyImage = load_grayscale_image(inputFile);
    cleanImage = load_grayscale_image(cleanFile);

    denoiseOpts = struct();
    denoiseOpts.dt = 0.20;
    denoiseOpts.maxIter = 25;
    denoiseOpts.tol = 1e-5;
    denoiseOpts.pcgTol = 1e-6;
    denoiseOpts.pcgMaxIter = 200;
    denoiseOpts.hx = 1.0;
    denoiseOpts.hy = 1.0;
    denoiseOpts.verbose = false;

    searchOpts = struct();
    searchOpts.lambdaCandidates = logspace(log10(0.01), log10(0.8), 7);
    searchOpts.KCandidates = logspace(log10(0.02), log10(0.20), 7);
    searchOpts.refinePasses = 3;
    searchOpts.refineLambdaPoints = 5;
    searchOpts.refineKPoints = 5;
    searchOpts.selectionCriterion = 'balanced';
    searchOpts.psnrWeight = 0.5;
    searchOpts.ssimWeight = 0.5;
    searchOpts.verbose = true;

    [u, best, history] = denoisingBDF2_Optimal_lambda_K(noisyImage, cleanImage, denoiseOpts, searchOpts);
    u = clip01(u);

    psnrNoisy = compute_psnr_basic(cleanImage, noisyImage);
    ssimNoisy = compute_ssim_basic(cleanImage, noisyImage);
    psnrDenoised = compute_psnr_basic(cleanImage, u);
    ssimDenoised = compute_ssim_basic(cleanImage, u);

    imwrite(u, outputImageFile);
    save_parameter_search_history(historyFile, history);
    write_summary_file_local(summaryFile, best, psnrNoisy, ssimNoisy, psnrDenoised, ssimDenoised, outputImageFile, historyFile);

    figure('Color', 'w', 'Name', 'Denoising BDF2 Demo - Adaptive lambda and K');

    subplot(2, 3, 1);
    imagesc(cleanImage);
    axis image off;
    colormap gray;
    title('Clean image');

    subplot(2, 3, 2);
    imagesc(noisyImage);
    axis image off;
    colormap gray;
    title(sprintf('Noisy image\nPSNR = %.2f dB\nSSIM = %.4f', psnrNoisy, ssimNoisy));

    subplot(2, 3, 3);
    imagesc(u);
    axis image off;
    colormap gray;
    title(sprintf('Adaptive \lambda,K result\nPSNR = %.2f dB\nSSIM = %.4f', psnrDenoised, ssimDenoised));

    subplot(2, 3, 4);
    scatter(history.lambda, history.K, 60, history.psnr, 'filled');
    hold on;
    plot(best.lambda, best.K, 'kp', 'MarkerSize', 12, 'MarkerFaceColor', 'y');
    hold off;
    set(gca, 'XScale', 'log', 'YScale', 'log');
    grid on;
    xlabel('\lambda');
    ylabel('K');
    title('Search grid colored by PSNR');
    colorbar;

    subplot(2, 3, 5);
    scatter(history.lambda, history.K, 60, history.ssim, 'filled');
    hold on;
    plot(best.lambda, best.K, 'kp', 'MarkerSize', 12, 'MarkerFaceColor', 'y');
    hold off;
    set(gca, 'XScale', 'log', 'YScale', 'log');
    grid on;
    xlabel('\lambda');
    ylabel('K');
    title('Search grid colored by SSIM');
    colorbar;

    subplot(2, 3, 6);
    axis off;
    text(0, 0.98, sprintf(['Selected \lambda = %.4f\n' ...
        'Selected K = %.4f\n' ...
        'PSNR = %.2f dB\n' ...
        'SSIM = %.4f\n\n' ...
        'Best-PSNR pair:\n  \lambda = %.4f, K = %.4f\n\n' ...
        'Best-SSIM pair:\n  \lambda = %.4f, K = %.4f'], ...
        best.lambda, best.K, psnrDenoised, ssimDenoised, ...
        best.bestByPSNR.lambda, best.bestByPSNR.K, ...
        best.bestBySSIM.lambda, best.bestBySSIM.K), ...
        'VerticalAlignment', 'top');

    fprintf('\nFinished adaptive joint (lambda, K) demo.\n');
    fprintf('Selected pair (%s): lambda = %.6f, K = %.6f\n', best.criterion, best.lambda, best.K);
    fprintf('Selected result PSNR = %.4f dB, SSIM = %.6f\n', psnrDenoised, ssimDenoised);
    fprintf('Best PSNR-only pair: lambda = %.6f, K = %.6f (PSNR = %.4f dB, SSIM = %.6f)\n', ...
        best.bestByPSNR.lambda, best.bestByPSNR.K, best.bestByPSNR.psnr, best.bestByPSNR.ssim);
    fprintf('Best SSIM-only pair: lambda = %.6f, K = %.6f (PSNR = %.4f dB, SSIM = %.6f)\n', ...
        best.bestBySSIM.lambda, best.bestBySSIM.K, best.bestBySSIM.psnr, best.bestBySSIM.ssim);
    fprintf('Saved denoised image to:\n%s\n', outputImageFile);
    fprintf('Saved parameter-search history to:\n%s\n', historyFile);
    fprintf('Saved summary to:\n%s\n', summaryFile);
end

function write_summary_file_local(filename, best, psnrNoisy, ssimNoisy, psnrDenoised, ssimDenoised, outputImageFile, historyFile)
%% WRITE_SUMMARY_FILE_LOCAL -   Save summary of the adaptive search.

    fid = fopen(filename, 'w');
    if fid == -1
        error('Could not open summary file for writing: %s', filename);
    end

    cleanupObj = onCleanup(@() fclose(fid)); 

    fprintf(fid, 'Adaptive lambda-K Denoising BDF2 summary\n');
    fprintf(fid, '===========================================\n\n');
    fprintf(fid, 'Selection criterion: %s\n', best.criterion);
    fprintf(fid, 'Selected lambda: %.12g\n', best.lambda);
    fprintf(fid, 'Selected K: %.12g\n', best.K);
    fprintf(fid, 'Selected result PSNR: %.12g dB\n', psnrDenoised);
    fprintf(fid, 'Selected result SSIM: %.12g\n\n', ssimDenoised);
    fprintf(fid, 'Noisy image PSNR: %.12g dB\n', psnrNoisy);
    fprintf(fid, 'Noisy image SSIM: %.12g\n\n', ssimNoisy);
    fprintf(fid, 'Best PSNR-only lambda: %.12g\n', best.bestByPSNR.lambda);
    fprintf(fid, 'Best PSNR-only K: %.12g\n', best.bestByPSNR.K);
    fprintf(fid, 'Best PSNR: %.12g dB\n', best.bestByPSNR.psnr);
    fprintf(fid, 'SSIM at best PSNR pair: %.12g\n\n', best.bestByPSNR.ssim);
    fprintf(fid, 'Best SSIM-only lambda: %.12g\n', best.bestBySSIM.lambda);
    fprintf(fid, 'Best SSIM-only K: %.12g\n', best.bestBySSIM.K);
    fprintf(fid, 'Best SSIM: %.12g\n', best.bestBySSIM.ssim);
    fprintf(fid, 'PSNR at best SSIM pair: %.12g dB\n\n', best.bestBySSIM.psnr);
    fprintf(fid, 'Number of parameter evaluations: %d\n\n', best.evaluations);
    fprintf(fid, 'Denoised image file: %s\n', outputImageFile);
    fprintf(fid, 'Parameter search history file: %s\n', historyFile);
end
