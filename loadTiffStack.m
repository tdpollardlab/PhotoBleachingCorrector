function [finalImage, height, width, numImages, tifData, fileName, pathName] = loadTiffStack(fileName,pathName,zSlices,colorChannels)
warning('off','all') %Used to suppress warnings resulting from black-and-white images having nonsense color codes or something

%Dealing with fewer input arguments
if nargin == 0
    [fileName,pathName] = uigetfile('.tif');
end
if nargin < 3
    zSlices = 1;
end
if nargin < 4
    colorChannels = 1;
end

%Determining properties of image (used instead of TIFF data because I
%suspect this is more universal)

imgInfo = imfinfo([pathName fileName]);
width = imgInfo(1).Width;
height = imgInfo(1).Height;
tifToLoad = Tiff([pathName fileName],'r');
tifTagNames = tifToLoad.getTagNames; %This lists all possible tags supported for the TIFF file (i.e. anything you could write)
tifData = struct;
for i = 1:length(tifTagNames) %Reads all of the possible tags that are actually present, to preserve image metadata
    try 
        tifData.(tifTagNames{i}) = tifToLoad.getTag(tifTagNames{i});
    catch
    end
    if strcmp(tifTagNames{i},'ImageDepth') && (nargin < 3 || zSlices == 0)
        zSlices = tifData.ImageDepth;
    end
end
numImages = numel(imgInfo)/(zSlices*colorChannels);
if floor(numImages) < numImages
    error('Number of images not divisible by number of Z-slices.');
end

finalImage = zeros(height,width,zSlices,numImages,colorChannels);

for t = 1:numImages
    for z = 1:zSlices
        for c = 1:colorChannels
            tifToLoad.setDirectory((t-1)*zSlices+z);
            finalImage(:,:,z,t,c) = tifToLoad.read();
        end
    end
end
tifToLoad.close();
warning('on','all')

