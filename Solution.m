%% Question 1:

% Implementation:
image = imread('Lev_noisysquares.png');
width = 3;
length = 3;
g = localhisteq(image, width, length);
figure()
subplot(2,2,1)
imshow(image)
title("Original image")
subplot(2,2,2)
bar(imhist(image))
title("Original histogram");
subplot(2,2,3)
imshow(g)
title("Image after local histogram")
subplot(2,2,4)
bar(imhist(g))
title("SumHist")




% Localize hist equals:
function g = localhisteq(I, M, N)
    %LOCALHISTEQ Local histogram equalization.
    % G = LOCALHISTEQ(I, M, N) perIorms local histogram equalization
    % on input image I using a window of (odd) size M-by-N to produce
    % the processed image, G. To handle border effects, the margins of I are cropped on
    % both sides by a width p, where M = 2*p + 1, and on the top and bottom by a width q , N
    % = 2*q+1 . [Optional - to handle border effects, image I is
    % extended by using the symmetric option in function padarray.
    % The amount of extension is determined by the dimensions of the
    % local window.] Both must be odd.
    %
    % This function accepts input images of class uint8, uint16, or
    % double. However, all computations are done using 8-bit intensity
    % values to speed-up computations. If I is of class double its
    % values should be in the range [0 1]. The class of the output
    % image is the same as the class of the input.
    
    % Check if the image type is valid.
    if not(isa(I,"double") || isa(I,"uint8") || isa(I, "uint16") || isa(I, "single") || isa(I, "double"))
        disp('There is no such type like this! please change it.')
    end

    % The calculation need to be in uint8:
    ConvertIToUint8 = im2uint8(I);
    
    [row, column] = size(ConvertIToUint8);

    % Take care of the margins by the window dimensions:
    p = (M-1)/2;    % Define the margin in above and in the bottom.
    q = (N-1)/2;    % Define the margin in left and in the right.
    
    
    
    % Padarray in the margins:
    IAfterPadarray = padarray(ConvertIToUint8, [p q], 0);

    % In order the pixels will not be bright too much on each other.
    SumHist = zeros(row,column);

    for i=p+1:row
        for j=q+1:column
            % For mirkuz on pixel (i,j):
            Rect = IAfterPadarray(i-p:i+p,j-q:j+q); % This is the window size M=2p+1, N=2q+1
            % Local histogram on the rect:
            NirmulFactor = sum(Rect);
            RectHist = imhist(Rect);
            LocalizeHistPixel = cumsum((RectHist*255)./NirmulFactor);
            LocalizeHistPixelNorm = round(LocalizeHistPixel);
            % Substitude the value in the window.
            if(Rect(p+1,q+1)==0)
                SumHist(i-p,j-q)=0;            
            else
                SumHist(i-p,j-q) = LocalizeHistPixelNorm(Rect(p+1,q+1));
            end
        end
    end
    g = SumHist;
end

