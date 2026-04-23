function save_parameter_search_history(filename, history)
%% SAVE_PARAMETER_SEARCH_HISTORY    -   Save joint values (lambda, K) search history as CSV.

    fid = fopen(filename, 'w');
    if fid == -1
        error('Could not open file for writing: %s', filename);
    end

    cleanupObj = onCleanup(@() fclose(fid)); 

    fprintf(fid, 'pass,lambda,K,psnr,ssim,score\n');
    for i = 1:numel(history.lambda)
        fprintf(fid, '%d,%.16g,%.16g,%.16g,%.16g,%.16g\n', ...
            history.pass(i), history.lambda(i), history.K(i), ...
            history.psnr(i), history.ssim(i), history.score(i));
    end
end
