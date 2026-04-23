function [uBest, best, history] = denoisingBDF2_Optimal_lambda_K(f, ref, denoiseOpts, searchOpts)
%%  DENOISINGBDF2_OPTIMAL_LAMBDA_K  -   Denoising using optimal values of lambda and K.
%
%   [uBest, best, history] = denoisingBDF2_Optimal_lambda_K(f, ref) searches for the pair (lambda, K) that 
%                            gives the best denoising results according to PSNR and SSIM measured against 
%                            the given clean reference images.
%
%   Inputs arguments:
%       f           -   Noisy grayscale image 
%       ref         -   Clean reference image of the same size as f
%       denoiseOpts -   Options passed to denoisingBDF2
%       searchOpts  -   Struct with optional fields:
%                           .lambdaCandidates    -  initial lambda grid
%                           .KCandidates         -  initial K grid
%                           .refinePasses        -  number of search passes (default 3)
%                           .refineLambdaPoints  -  points in each lambda refinement pass
%                           .refineKPoints       -  points in each K refinement pass
%                           .selectionCriterion  -  'balanced', 'psnr', or 'ssim'
%                           .psnrWeight          -  weight for PSNR in balanced mode
%                           .ssimWeight          -  weight for SSIM in balanced mode
%                           .verbose             -  display tuning progress (default true)
%
%   Additional information:
%   - Joint maximization of PSNR and SSIM requires a clean reference image.
%   - In balanced mode (eqn. (2.20) in the paper), the selected pair is the candidate closest to the
%     ideal point formed by the best normalized PSNR and SSIM values.
%   - The function performs a coarse-to-fine heuristic logarithmic search in both
%     lambda and K, and returns the best pair found among all evaluations.

    if nargin < 3 || isempty(denoiseOpts)
        denoiseOpts = struct();
    end
    if nargin < 4 || isempty(searchOpts)
        searchOpts = struct();
    end

    if ~ismatrix(f) || ~ismatrix(ref)
        error('Both f and ref must be 2D grayscale images.');
    end
    if ~isequal(size(f), size(ref))
        error('Both noisy and reference images must have the same size.');
    end

    f = normalize_to_unit_interval_local(f);
    ref = normalize_to_unit_interval_local(ref);

    searchOpts = set_default_local(searchOpts, 'lambdaCandidates', logspace(log10(0.01), log10(0.8), 7));
    searchOpts = set_default_local(searchOpts, 'KCandidates', logspace(log10(0.02), log10(0.20), 7));
    searchOpts = set_default_local(searchOpts, 'refinePasses', 3);
    searchOpts = set_default_local(searchOpts, 'refineLambdaPoints', 5);
    searchOpts = set_default_local(searchOpts, 'refineKPoints', 5);
    searchOpts = set_default_local(searchOpts, 'selectionCriterion', 'balanced');
    searchOpts = set_default_local(searchOpts, 'psnrWeight', 0.5);
    searchOpts = set_default_local(searchOpts, 'ssimWeight', 0.5);
    searchOpts = set_default_local(searchOpts, 'verbose', true);

    lambdaGrid = validate_positive_list(searchOpts.lambdaCandidates, 'lambda');
    KGrid = validate_positive_list(searchOpts.KCandidates, 'K');

    if searchOpts.refinePasses < 1 || round(searchOpts.refinePasses) ~= searchOpts.refinePasses
        error('refinePasses must be a positive integer.');
    end
    if searchOpts.refineLambdaPoints < 3 || round(searchOpts.refineLambdaPoints) ~= searchOpts.refineLambdaPoints
        error('refineLambdaPoints must be an integer >= 3.');
    end
    if searchOpts.refineKPoints < 3 || round(searchOpts.refineKPoints) ~= searchOpts.refineKPoints
        error('refineKPoints must be an integer >= 3.');
    end

    allLambda = [];
    allK = [];
    allPSNR = [];
    allSSIM = [];
    allScore = [];
    allPass = [];
    allImages = {};
    allInfos = {};

    currentLambdaGrid = lambdaGrid;
    currentKGrid = KGrid;

    if searchOpts.verbose
        fprintf('Adaptive (lambda, K) search started.\n');
        fprintf('Selection criterion: %s\n', searchOpts.selectionCriterion);
    end

    for pass = 1:searchOpts.refinePasses
        currentLambdaGrid = validate_positive_list(currentLambdaGrid, 'lambda');
        currentKGrid = validate_positive_list(currentKGrid, 'K');

        nLambda = numel(currentLambdaGrid);
        nK = numel(currentKGrid);
        nCand = nLambda * nK;

        if searchOpts.verbose
            fprintf('\nPass %d/%d\n', pass, searchOpts.refinePasses);
            fprintf('  lambda candidates: ');
            disp(currentLambdaGrid(:).');
            fprintf('  K candidates:      ');
            disp(currentKGrid(:).');
            fprintf('  total evaluations this pass: %d\n', nCand);
        end

        passLambda = zeros(nCand, 1);
        passK = zeros(nCand, 1);
        passPSNR = zeros(nCand, 1);
        passSSIM = zeros(nCand, 1);
        passImages = cell(nCand, 1);
        passInfos = cell(nCand, 1);

        idxCand = 0;
        for i = 1:nLambda
            for j = 1:nK
                idxCand = idxCand + 1;

                optsLocal = denoiseOpts;
                optsLocal.lambda = currentLambdaGrid(i);
                optsLocal.K = currentKGrid(j);
                optsLocal.verbose = false;

                [uCand, infoCand] = denoisingBDF2(f, optsLocal);
                psnrVal = compute_psnr_basic(ref, uCand);
                ssimVal = compute_ssim_basic(ref, uCand);

                passLambda(idxCand) = currentLambdaGrid(i);
                passK(idxCand) = currentKGrid(j);
                passPSNR(idxCand) = psnrVal;
                passSSIM(idxCand) = ssimVal;
                passImages{idxCand} = uCand;
                passInfos{idxCand} = infoCand;

                if searchOpts.verbose
                    fprintf('  lambda = %.6f, K = %.6f -> PSNR = %.4f dB, SSIM = %.6f\n', ...
                        currentLambdaGrid(i), currentKGrid(j), psnrVal, ssimVal);
                end
            end
        end

        passScore = compute_selection_score_local(passPSNR, passSSIM, ...
                                                  searchOpts.selectionCriterion, searchOpts.psnrWeight, searchOpts.ssimWeight);
        passBestIdx = select_best_index_local(passPSNR, passSSIM, passScore, searchOpts.selectionCriterion);

        allLambda = [allLambda; passLambda]; %#ok<AGROW> - "%#ok<AGROW>" Supresses warnings about array growth within the loop
        allK = [allK; passK]; %#ok<AGROW>
        allPSNR = [allPSNR; passPSNR]; %#ok<AGROW>
        allSSIM = [allSSIM; passSSIM]; %#ok<AGROW>
        allScore = [allScore; passScore]; %#ok<AGROW>
        allPass = [allPass; pass * ones(nCand, 1)]; %#ok<AGROW>
        allImages = [allImages; passImages]; %#ok<AGROW>
        allInfos = [allInfos; passInfos]; %#ok<AGROW>

        if searchOpts.verbose
            fprintf('Best pair in pass %d: lambda = %.6f, K = %.6f\n', ...
                pass, passLambda(passBestIdx), passK(passBestIdx));
        end

        if pass < searchOpts.refinePasses
            bestLambdaThisPass = passLambda(passBestIdx);
            bestKThisPass = passK(passBestIdx);
            bestLambdaIdx = find(abs(currentLambdaGrid - bestLambdaThisPass) <= 1e-14 * max(1, abs(bestLambdaThisPass)), 1, 'first');
            bestKIdx = find(abs(currentKGrid - bestKThisPass) <= 1e-14 * max(1, abs(bestKThisPass)), 1, 'first');

            currentLambdaGrid = refine_positive_grid(currentLambdaGrid, bestLambdaIdx, searchOpts.refineLambdaPoints);
            currentKGrid = refine_positive_grid(currentKGrid, bestKIdx, searchOpts.refineKPoints);
        end
    end

    overallScore = compute_selection_score_local(allPSNR, allSSIM, ...
                                                 searchOpts.selectionCriterion, searchOpts.psnrWeight, searchOpts.ssimWeight);
    bestIdx = select_best_index_local(allPSNR, allSSIM, overallScore, searchOpts.selectionCriterion);

    uBest = allImages{bestIdx};

    [maxPSNR, idxPSNR] = max(allPSNR);
    [maxSSIM, idxSSIM] = max(allSSIM);

    best = struct();
    best.lambda = allLambda(bestIdx);
    best.K = allK(bestIdx);
    best.psnr = allPSNR(bestIdx);
    best.ssim = allSSIM(bestIdx);
    best.score = overallScore(bestIdx);
    best.criterion = searchOpts.selectionCriterion;
    best.info = allInfos{bestIdx};

    best.bestByPSNR = struct();
    best.bestByPSNR.lambda = allLambda(idxPSNR);
    best.bestByPSNR.K = allK(idxPSNR);
    best.bestByPSNR.psnr = maxPSNR;
    best.bestByPSNR.ssim = allSSIM(idxPSNR);

    best.bestBySSIM = struct();
    best.bestBySSIM.lambda = allLambda(idxSSIM);
    best.bestBySSIM.K = allK(idxSSIM);
    best.bestBySSIM.psnr = allPSNR(idxSSIM);
    best.bestBySSIM.ssim = maxSSIM;

    best.evaluations = numel(allLambda);

    history = struct();
    history.lambda = allLambda;
    history.K = allK;
    history.psnr = allPSNR;
    history.ssim = allSSIM;
    history.score = overallScore;
    history.pass = allPass;
    history.selectionCriterion = searchOpts.selectionCriterion;
    history.psnrWeight = searchOpts.psnrWeight;
    history.ssimWeight = searchOpts.ssimWeight;

    if searchOpts.verbose
        fprintf('\nAdaptive (lambda, K) search finished.\n');
        fprintf('Selected pair (%s): lambda = %.6f, K = %.6f\n', ...
            searchOpts.selectionCriterion, best.lambda, best.K);
        fprintf('Selected result metrics: PSNR = %.4f dB, SSIM = %.6f\n', best.psnr, best.ssim);
        fprintf('Best PSNR-only pair: lambda = %.6f, K = %.6f (PSNR = %.4f dB, SSIM = %.6f)\n', ...
            best.bestByPSNR.lambda, best.bestByPSNR.K, best.bestByPSNR.psnr, best.bestByPSNR.ssim);
        fprintf('Best SSIM-only pair: lambda = %.6f, K = %.6f (PSNR = %.4f dB, SSIM = %.6f)\n', ...
            best.bestBySSIM.lambda, best.bestBySSIM.K, best.bestBySSIM.psnr, best.bestBySSIM.ssim);
    end
end

function x = normalize_to_unit_interval_local(x)
%%  NORMALIZE_TO_UNIT_INTERVAL_LOCAL    - Normalization of numeric image data to a range within [0, 1].

    x = double(x);
    xmin = min(x(:));
    xmax = max(x(:));

    if xmax > 1 || xmin < 0
        if xmax > xmin
            x = (x - xmin) / (xmax - xmin);
        else
            x = zeros(size(x));
        end
    end

    x = clip01(x);
end

function vals = validate_positive_list(vals, nameStr)
%% VALIDATE_POSITIVE_LIST   -   Ensure that the candidate values are positive and sorted.

    vals = double(vals(:));
    vals = vals(isfinite(vals) & vals > 0);
    vals = unique(vals(:).');

    if isempty(vals)
        error('At least one positive %s candidate must be supplied.', nameStr);
    end
end

function score = compute_selection_score_local(psnrVals, ssimVals, criterion, wPSNR, wSSIM)
% COMPUTE_SELECTION_SCORE_LOCAL -   Compute the BalancedSscore (eqn. (2.18)) used to select the best pair.

    criterion = lower(strtrim(criterion));

    switch criterion
        case 'psnr'
            score = psnrVals;
        case 'ssim'
            score = ssimVals;
        case 'balanced'
            psnrNorm = safe_unit_normalize_local(psnrVals);
            ssimNorm = safe_unit_normalize_local(ssimVals);
            dist = sqrt(wPSNR * (1 - psnrNorm) .^ 2 + wSSIM * (1 - ssimNorm) .^ 2);
            score = -dist;
        otherwise
            error('Unknown selection criterion: %s', criterion);
    end
end

function idx = select_best_index_local(psnrVals, ssimVals, score, criterion)
%% SELECT_BEST_INDEX_LOCAL -    Choose the best candidate with deterministic tie-breaking.

    criterion = lower(strtrim(criterion));

    switch criterion
        case {'psnr', 'ssim', 'balanced'}
            bestValue = max(score);
            candidates = find(abs(score - bestValue) <= 1e-12 * max(1, abs(bestValue)));
            if numel(candidates) == 1
                idx = candidates;
                return;
            end

            subSSIM = ssimVals(candidates);
            maxSSIM = max(subSSIM);
            candidates2 = candidates(abs(subSSIM - maxSSIM) <= 1e-12 * max(1, abs(maxSSIM)));

            if numel(candidates2) == 1
                idx = candidates2;
                return;
            end

            subPSNR = psnrVals(candidates2);
            [~, localIdx] = max(subPSNR);
            idx = candidates2(localIdx(1));
        otherwise
            error('Unknown selection criterion: %s', criterion);
    end
end

function valsNorm = safe_unit_normalize_local(vals)
%% SAFE_UNIT_NORMALIZE_LOCAL -  Normalize a vector to within a range of [0, 1] with robust handling.

    vals = double(vals(:));
    valsNorm = zeros(size(vals));

    isFinite = isfinite(vals);
    if ~any(isFinite)
        valsNorm(:) = 1;
        return;
    end

    finiteVals = vals(isFinite);
    vmin = min(finiteVals);
    vmax = max(finiteVals);

    if vmax > vmin
        valsNorm(isFinite) = (vals(isFinite) - vmin) / (vmax - vmin);
    else
        valsNorm(isFinite) = 1;
    end

    valsNorm(~isFinite) = 1;
end

function newGrid = refine_positive_grid(currentGrid, bestIdx, nPoints)
%% REFINE_POSITIVE_GRID -   Create a refined logarithmic grid around the current best.

    currentGrid = validate_positive_list(currentGrid, 'parameter');
    currentGrid = sort(currentGrid(:));
    bestVal = currentGrid(bestIdx);

    if numel(currentGrid) == 1
        left = bestVal / 2;
        right = bestVal * 2;
    elseif bestIdx == 1
        left = max(bestVal / 4, 1e-6);
        right = currentGrid(2);
    elseif bestIdx == numel(currentGrid)
        left = currentGrid(end - 1);
        right = currentGrid(end) * 4;
    else
        left = currentGrid(bestIdx - 1);
        right = currentGrid(bestIdx + 1);
    end

    left = max(left, 1e-6);
    right = max(right, left * 1.0001);

    newGrid = logspace(log10(left), log10(right), nPoints);
    newGrid = validate_positive_list([bestVal, newGrid], 'parameter');
end

function s = set_default_local(s, name, value)
%% SET_DEFAULT_LOCAL -  Set a default value if a field is missing or empty.

    if ~isfield(s, name) || isempty(s.(name))
        s.(name) = value;
    end
end
