%% M-File Codes for adding White Gaussian Noise
img = imread('data/03.png'); % Read original (clean) image from the "data/..." source folder containing images
I = double(img); % Convert the image into "double" class
sigma = 25; % Standard deviation of noise, AWGN
snr_db = 10*log10(mean(I(:).^2)/(sigma^2));
J = awgn(I, snr_db, 'measured');
J = uint8(min(max(J,0),255));
imwrite(J,'data/03Noise.png','png');