function f = load_grayscale_image(filename)
%% LOAD_GRAYSCALE_IMAGE -   Read an image and convert it to grayscale double with entries within a range of [0, 1].

    if ~exist(filename, 'file')
        error('File not found: %s', filename);
    end

    img = imread(filename);
    imgClass = class(img);
    imgDouble = double(img);

    if ndims(imgDouble) == 3
        imgDouble = 0.2989 * imgDouble(:, :, 1) + ...
                    0.5870 * imgDouble(:, :, 2) + ...
                    0.1140 * imgDouble(:, :, 3);
    end

    switch imgClass
        case 'uint8'
            f = imgDouble / 255;
        case 'uint16'
            f = imgDouble / 65535;
        otherwise
            vmin = min(imgDouble(:));
            vmax = max(imgDouble(:));
            if vmax > vmin
                if vmax > 1 || vmin < 0
                    f = (imgDouble - vmin) / (vmax - vmin);
                else
                    f = imgDouble;
                end
            else
                f = zeros(size(imgDouble));
            end
    end

    f = clip01(f);
end
