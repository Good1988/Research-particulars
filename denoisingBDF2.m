function [u, info] = denoisingBDF2(f, opts)
%% DENOISINGBDF2  - Denoising an image using second-order backward differentiation formula.
%  Nonlinear diffusion equation:
%       u_t = div(phi(|grad u|) grad u) - lambda * (u - f)
%       phi(s) = 1 / (1 + (s / K)^4)  [phi(s) can be replaced by another
%                                      diffusion coefficient]
%       du/dn = 0 on the boundary [ Newmann conditions]
%
%   Usage:
%       [u, info] = denoisingBDF2(f);
%       [u, info] = denoisingBDF2(f, opts);
%
%   Input arguments:
%       f    - Scaled 2D grayscale image (image scaled to [0, 1])
%       opts - struct (data type container) with the following optional fields:
%              .dt         -    Time step size; default: 0.20
%              .lambda     -    Fidelity weight; default: 0.12
%              .K          -    Contrast parameter; default: 0.08
%              .maxIter    -    Number of time steps; default: 40
%              .tol        -    Stopping tolerance; default: 1e-5
%              .pcgTol     -    PCG (Preconditioned Conjugate Gradient) tolerance; default: 1e-6
%              .pcgMaxIter -    PCG max iterations; default: 200
%              .hx         -    Grid spacing along the horizontal direction, x; default: 1.0
%              .hy         -    Grid spacing along the vertical direction, y; default: 1.0
%              .verbose    -    Display iteration log; default: true
%
%   Output arguments:
%       u    - denoised image
%       info - struct (data type) with run information
%
%   Additional Information:
%   - The first time step uses Backward Euler method
%   - The subsequent time steps use BDF2 method
%   - The nonlinear diffusivity component is evaluated from the previous
%     iteration, which produces a linear sparse system for each time step
%   - This implementation does not need any external toolbox

    if nargin < 2
        opts = struct();
    end

% ================== Validation of the Input Image ========================
    if ~ismatrix(f)
        error('The input image, f, must be a two-dimensional (2D) grayscale image.');
    end
    if isempty(f)
        error('The input image, f, should NOT be empty.');
    end

    f = double(f);

% Normalization of the input image, f, if its elements are not within [0, 1].
    fmin = min(f(:));
    fmax = max(f(:));
    if fmax > 1 || fmin < 0
        if fmax > fmin
            f = (f - fmin) / (fmax - fmin);
        else
            f = zeros(size(f));
        end
    end

% =========================== DEFAULT SETTINGS ============================
%   .dt          -   Time step size: 0.20
%   .lambda      -   Fidelity weight: 0.12
%   .K           -   Gradient-thresholding parameter: 0.08
%   .maxIter     -   Number of time steps: 40
%   .tol         -   Stopping tolerance: 1e-5
%   .pcgTol      -   PCG (Preconditioned Conjugate Gradient) tolerance: 1e-6
%   .pcgMaxIter  -   PCG max iterations: 200
%   .hx          -   Grid spacing along the horizontal direction, x: 1.0
%   .hy          -   Grid spacing along the vertical direction, y: 1.0
%   .verbose     -   Display iteration log: true

    opts = set_default(opts, 'dt', 0.20);
    opts = set_default(opts, 'lambda', 0.12);
    opts = set_default(opts, 'K', 0.08);
    opts = set_default(opts, 'maxIter', 40);
    opts = set_default(opts, 'tol', 1e-5);
    opts = set_default(opts, 'pcgTol', 1e-6);
    opts = set_default(opts, 'pcgMaxIter', 200);
    opts = set_default(opts, 'hx', 1.0);
    opts = set_default(opts, 'hy', 1.0);
    opts = set_default(opts, 'verbose', true);

    if opts.dt <= 0
        error('dt must be positive.');
    end
    if opts.lambda < 0
        error('lambda must be nonnegative.');
    end
    if opts.K <= 0
        error('K must be positive.');
    end
    if opts.maxIter < 1 || round(opts.maxIter) ~= opts.maxIter
        error('maxIter must be a positive integer.');
    end
    if opts.tol <= 0
        error('tol must be positive.');
    end
    if opts.pcgTol <= 0
        error('pcgTol must be positive.');
    end
    if opts.pcgMaxIter < 1 || round(opts.pcgMaxIter) ~= opts.pcgMaxIter
        error('pcgMaxIter must be a positive integer.');
    end
    if opts.hx <= 0 || opts.hy <= 0
        error('hx and/or hy must be positive.');
    end

    [m, n] = size(f); % Get the size of the input image, f: m = number of rows; n = number of columns
    % N = m * n;

    dt = opts.dt;
    lambda = opts.lambda;
    K = opts.K;
    hx = opts.hx;
    hy = opts.hy;

% ============================ Initialization =============================
    u0 = f;
    relChanges = nan(opts.maxIter, 1);
    pcgFlags = nan(opts.maxIter, 1);
    pcgRelres = nan(opts.maxIter, 1);
    pcgIters = nan(opts.maxIter, 1);

    if opts.verbose
        fprintf('Running BDF2 nonlinear diffusion denoising on %d x %d image\n', m, n);
        fprintf('dt = %.4f, lambda = %.4f, K = %.4f, maxIter = %d\n', dt, lambda, K, opts.maxIter);
    end

% =========================== First step: Backward Euler ==================
    phi0 = compute_diffusivity(u0, K, hx, hy);
    alpha1 = 1 / dt + lambda;
    A1 = build_system_matrix(phi0, alpha1, hx, hy);
    b1 = u0(:) / dt + lambda * f(:);

    [u1, flag, relres, iter] = solve_spd_system(A1, b1, [m, n], opts.pcgTol, opts.pcgMaxIter);
    pcgFlags(1) = flag;
    pcgRelres(1) = relres;
    pcgIters(1) = iter;

    relChanges(1) = norm(u1(:) - u0(:), 2) / max(norm(u0(:), 2), eps);

    if opts.verbose
        fprintf('Step %3d (BE)  relChange = %.6e, pcgFlag = %d, pcgRelres = %.3e, pcgIter = %d\n', ...
            1, relChanges(1), flag, relres, iter);
    end

    if opts.maxIter == 1 || relChanges(1) < opts.tol
        u = u1;
        info = pack_info(1, relChanges, pcgFlags, pcgRelres, pcgIters);
        return;
    end

 % ================ Subsequent steps: BDF2 Numerical Scheme================
    u_prev = u0;
    u_curr = u1;
    finalIter = 1;

    for k = 2:opts.maxIter
        phi = compute_diffusivity(u_curr, K, hx, hy);

        alpha = 3 / (2 * dt) + lambda;
        A = build_system_matrix(phi, alpha, hx, hy);
        b = (4 * u_curr(:) - u_prev(:)) / (2 * dt) + lambda * f(:);

        [u_next, flag, relres, iter] = solve_spd_system(A, b, [m, n], opts.pcgTol, opts.pcgMaxIter);
        pcgFlags(k) = flag;
        pcgRelres(k) = relres;
        pcgIters(k) = iter;

        relChanges(k) = norm(u_next(:) - u_curr(:), 2) / max(norm(u_curr(:), 2), eps);
        finalIter = k;

        if opts.verbose
            fprintf('Step %3d (BDF2) relChange = %.6e, pcgFlag = %d, pcgRelres = %.3e, pcgIter = %d\n', ...
                k, relChanges(k), flag, relres, iter);
        end

        u_prev = u_curr;
        u_curr = u_next;

        if relChanges(k) < opts.tol
            break;
        end
    end

    u = u_curr;
    info = pack_info(finalIter, relChanges, pcgFlags, pcgRelres, pcgIters);
end

function phi = compute_diffusivity(u, K, hx, hy)
%% COMPUTE_DIFFUSIVITY  -   Compute phi(|grad u|) using mirrored Neumann padding.

    up = pad_neumann(u);

    ux = (up(2:end-1, 3:end) - up(2:end-1, 1:end-2)) / (2 * hx);
    uy = (up(3:end, 2:end-1) - up(1:end-2, 2:end-1)) / (2 * hy);

    s = sqrt(ux .^ 2 + uy .^ 2);
    phi = 1 ./ (1 + (s ./ K) .^ 4);
end

function up = pad_neumann(u)
%% PAD_NEUMANN  -   Mirror the nearest boundary value to impose zero normal flux.

    [m, n] = size(u);

    up = zeros(m + 2, n + 2);
    up(2:end-1, 2:end-1) = u;

    up(1, 2:end-1) = u(1, :);
    up(end, 2:end-1) = u(end, :);
    up(2:end-1, 1) = u(:, 1);
    up(2:end-1, end) = u(:, end);

    up(1, 1) = u(1, 1);
    up(1, end) = u(1, end);
    up(end, 1) = u(end, 1);
    up(end, end) = u(end, end);
end

function A = build_system_matrix(phi, alpha, hx, hy)
%% BUILD_SYSTEM_MATRIX  -   Build sparse SPD system matrix
%   A = alpha*I - L(phi)
%   L(phi) gives the conservative diffusion operator with zero-flux boundary.

    [m, n] = size(phi);
    N = m * n;
    idx = reshape(1:N, m, n);

    % Face conductivities along the East (E), West (W), South (S), and North (N) directions
    phiE = zeros(m, n);
    phiW = zeros(m, n);
    phiS = zeros(m, n);
    phiN = zeros(m, n);

    if n > 1
        faceX = 0.5 * (phi(:, 1:end-1) + phi(:, 2:end)) / (hx * hx);
        phiE(:, 1:end-1) = faceX;
        phiW(:, 2:end) = faceX;
    end

    if m > 1
        faceY = 0.5 * (phi(1:end-1, :) + phi(2:end, :)) / (hy * hy);
        phiS(1:end-1, :) = faceY;
        phiN(2:end, :) = faceY;
    end

% Maximum number of nonzeros in a five-point stencil
    ii = zeros(5 * N, 1);
    jj = zeros(5 * N, 1);
    ss = zeros(5 * N, 1);
    ptr = 0;

    for col = 1:n
        for row = 1:m
            p = idx(row, col);

            center = alpha + phiE(row, col) + phiW(row, col) + phiS(row, col) + phiN(row, col);

            ptr = ptr + 1;
            ii(ptr) = p;
            jj(ptr) = p;
            ss(ptr) = center;

            if col < n
                q = idx(row, col + 1);
                ptr = ptr + 1;
                ii(ptr) = p;
                jj(ptr) = q;
                ss(ptr) = -phiE(row, col);
            end

            if col > 1
                q = idx(row, col - 1);
                ptr = ptr + 1;
                ii(ptr) = p;
                jj(ptr) = q;
                ss(ptr) = -phiW(row, col);
            end

            if row < m
                q = idx(row + 1, col);
                ptr = ptr + 1;
                ii(ptr) = p;
                jj(ptr) = q;
                ss(ptr) = -phiS(row, col);
            end

            if row > 1
                q = idx(row - 1, col);
                ptr = ptr + 1;
                ii(ptr) = p;
                jj(ptr) = q;
                ss(ptr) = -phiN(row, col);
            end
        end
    end

    A = sparse(ii(1:ptr), jj(1:ptr), ss(1:ptr), N, N);
end

function [u, flag, relres, iter] = solve_spd_system(A, b, outSize, pcgTol, pcgMaxIter)
%% SOLVE_SPD_SYSTEM -   Solve Ax = b using PCG (Preconditioned Conjugate Gradient) with a diagonal preconditioner.
%   The function falls back to the sparse direct solver if PCG fails to converge.

    d = diag(A);
    d = max(d, eps);
    M = spdiags(d, 0, size(A, 1), size(A, 2));

    [x, flag, relres, iter] = pcg(A, b, pcgTol, pcgMaxIter, M);

    if flag ~= 0 || any(~isfinite(x))
        warning('PCG fails converge (flag=%d, relres=%g). Falling back to backslash.', flag, relres);
        x = A \ b;
        flag = 0;
        relres = norm(A * x - b, 2) / max(norm(b, 2), eps);
        iter = 0;
    end

    u = reshape(x, outSize);
end

function opts = set_default(opts, name, value)
%% SET_DEFAULT  -   Set default options if missing.

    if ~isfield(opts, name) || isempty(opts.(name))
        opts.(name) = value;
    end
end

function info = pack_info(finalIter, relChanges, pcgFlags, pcgRelres, pcgIters)
%% PACK_INFO    -   Package output information.

    info = struct();
    info.iterations = finalIter;
    info.relChanges = relChanges(1:finalIter);
    info.pcgFlags = pcgFlags(1:finalIter);
    info.pcgRelres = pcgRelres(1:finalIter);
    info.pcgIters = pcgIters(1:finalIter);
end
