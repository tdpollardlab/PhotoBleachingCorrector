function differenceVector = GaussFit2D(x0,inputImage) %x0: a, h0, w0, sigmaH,sigmaW
[height,width] = size(inputImage);

A = x0(1);
h0 = x0(2);
w0 = x0(3);
sigmaH = x0(4);
sigmaW = x0(end); %4 if only 4 components, in which case both sigmas always the same

%Plotting the Gaussian
gaussImage = zeros(height,width);
for h = 1:height
    for w = 1:width
        gaussExponent = -(h-h0)^2/(2*sigmaH^2)-(w-w0)^2/(2*sigmaW^2);
        gaussImage(h,w) = A*exp(gaussExponent);
    end
end

%Subtracting the image and finding the total squared deviation (to be
%minimized)

%imageSquaredDifference = (inputImage - gaussImage).^2;
%squaredDifference = sum(sum(imageSquaredDifference);

%Finding vector deviation (apparently what they want);
differenceImage = double(inputImage)-double(gaussImage);
differenceVector = reshape(differenceImage,[height*width 1]); 