function out = clip01(in)
%% CLIP01 - Clip image intensities to the interval [0, 1].

    out = min(max(double(in), 0), 1);
end
