function save_lambda_search_history(filename, history)
%% SAVE_LAMBDA_SEARCH_HISTORY   - Save lambda search metrics to a CSV file.

    fid = fopen(filename, 'w');
    if fid == -1
        error('Could not open file for writing: %s', filename);
    end

    cleanupObj = onCleanup(@() fclose(fid));

    fprintf(fid, 'pass,lambda,psnr_db,ssim,score\n');
    for k = 1:numel(history.lambda)
        fprintf(fid, '%d,%.12g,%.12g,%.12g,%.12g\n', ...
            history.pass(k), history.lambda(k), history.psnr(k), history.ssim(k), history.score(k));
    end
end
