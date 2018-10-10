%% Automatic photobleaching corrector

close all

%% Inputting settings and loading folder with images

createFakeImages = false;
correctNoiseUneven = false;
createNoiseUneven = false;
automaticSegmentation = false;
saveCorrectedImages = false;
plotAllGraphs = true;

%Defaults
zSlices = 6;
curveType = 'exp2';
unevenFraction = 0.5; %percentage of max illumination that is OK
numFramesToSegment = 10; %number of frames to use for segmentation
doSegmentationTimes = 1; %how often segmentation is done (evenly spaced). Should be 1 if your cells do not move
overWrite = false;

prompt = {'Number of Z-slices:','Curve fitting model (options: none, exp1, exp2)','Enter 1 to create fake images for patch tracking program','Enter 1 to correct noise and uneven illumination','Enter 1 for automatic segmentation','Enter 1 to overwrite existing segmentations','Enter 1 to save corrected images','Enter 1 to overwrite images (originals saved under modified filename)','Fraction of uneven illumination image to keep (1 if all)','Number of frames to use for segmentation','Number of timepoints at which to perform segmentation','Enter 1 for advanced segmentation settings'};
dialogTitle = 'Input';
numLines=1;
defaultAns={num2str(zSlices),curveType,'1','1','1','0','1','0',num2str(unevenFraction),num2str(numFramesToSegment),num2str(doSegmentationTimes),'0'};
data = inputdlg(prompt,dialogTitle,numLines,defaultAns); 
zSlices = str2double(data(1)); %set to 0 for automatic
fitMethod = data{2}; %options: 'none','exp1','exp2'
if(str2double(data(3)) == 1)
    createFakeImages = true; %creates fake images for Julien's program to "correct"
end
if(str2double(data(4)) == 1)
    correctNoiseUneven = true;
end
if(str2double(data(5)) == 1)
    automaticSegmentation = true;
end
if(str2double(data(6)) == 1)
    newSegmentation = true;
end
if(str2double(data(7)) == 1)
    saveCorrectedImages = true;
end
if(str2double(data(8)) == 1)
    overWrite = true;
end
unevenFraction = str2double(data(9));
numFramesToSegment = str2double(data(10));
doSegmentationTimes = str2double(data(11));


%WARNING: Make sure your CameraNoise and UnevenIllumination images are
%named properly (see documentation)
%WARNING: This assumes you have already subtracted the camera noise from your
%uneven illumination image. If not, please do that in ImageJ.
%WARNING: Images with substantial uneven illumination or camera noise may
%not work as well.

%Advanced segmentation settings (change only if segmentation works poorly)
threshold = 0.95; %Sets cutoff along Gaussian
baseRescaleSize = 100; %rescaled width (pixels) for Gaussian fitting for uneven illumination
fractionToPass = 0.8; %Percentage of sampled frames in which pixels must be intracellular'
segmentationBinSize = 25;

if(str2double(data(12))==1) %check advanced settings
    prompt = {'Segmentation threshold (max 1)','Rescaled image width for pixel brightness histogram (warning: if too big, may crash program)','Fraction of test frames in which an intracellular pixel must be detected','Bin size for histogram used for segmentation'};
    dialogTitle = 'Advanced Settings';
    numLines = 1;
    defaultAns = {num2str(threshold),num2str(baseRescaleSize),num2str(fractionToPass),num2str(segmentationBinSize)};
    data2 = inputdlg(prompt,dialogTitle,numLines,defaultAns); 
    threshold = str2double(data2(1));
    baseRescaleSize = str2double(data2(2));
    fractionToPass = str2double(data2(3));
    segmentationBinSize = str2double(data2(4));
end

%NOTE: Cell segmentation will not be perfect. You should not be concerned
%unless your curve looks bad, and either >50% of intracellular space is
%missed or 50% of selected pixels are not actually in cells.

%Selecting main folder

fprintf('Select parent folder with patch movies \n');
dirInput = uigetdir('','Select parent folder with patch movies');

allFiles = dir(dirInput);
i = 1;
for j = 1:numel(allFiles) %to remove trash folders in Mac
    if allFiles(i).name(1) == '.'
        allFiles(i) = [];
    else
        i = i + 1;
    end
end

% if createNoiseUneven
%     [unevenTempName,unevenTempPath] = uigetfile('.tif','Select uneven illumination image');
%     [backgroundTempName,backgroundTempPath] = uigetfile('.tif','Select background image');
%     backgroundPath = [
%     [backgroundInput,height,width,numFrames,tifData] = loadTiffStack(fileName,background
% end

%% Loading uneven illumination and background noise, and segmenting uneven illumination to create mask

%load uneven illumination and background noise files. Can replace later
%with options to select them, but I actually prefer it this way.
if correctNoiseUneven
    %Check for existence of background file and create it if missing
    fprintf('Checking for camera noise and uneven illumination images. \n');
    
    if ~exist([dirInput '/CameraNoise.tif'],'file') %if it doesn't exist
        fprintf('Select camera noise image stack');
        cameraNoiseRaw = loadTiffStack; %if no arguments provided
        cameraNoise = mean(cameraNoiseRaw,3); %mean projection
        imwrite(cameraNoise,[dirInput '/CameraNoise.tif']);
    else
        cameraNoise = double(imread([dirInput '/CameraNoise.tif']));
    end

    if ~exist([dirInput '/UnevenIllumination.tif'],'file')
        fprintf('Select uneven illumination image stack');
        unevenIlluminationRaw = loadTiffStack;
        unevenIllumination = mean(unevenIlluminationRaw,3);
        unevenIllumination = unevenIllumination - cameraNoise;
        imwrite(unevenIllumination,[dirInput '/CameraNoise.tif']);
    else
        unevenIllumination = double(imread([dirInput '/UnevenIllumination.tif']));
        unevenMax = max(max(unevenIllumination));
        if unevenMax>1
            unevenIllumination = unevenIllumination./unevenMax;
        end
    end
    
    if exist([dirInput '/imageMask.tif'],'file') && unevenFraction < 1
        imageMask = imread([dirInput '/imageMask.tif']);
    elseif unevenFraction < 1
        imageMask = ones(size(unevenIllumination));
        %Process uneven illumination image and save it
        
        %Blurring image for easier Gaussian fit
        [unevenHeight,unevenWidth] = size(unevenIllumination);
        rescaleFactorBase = unevenWidth/baseRescaleSize; 
        unevenGCD = gcd(unevenHeight,unevenWidth);
        if unevenGCD >= rescaleFactorBase/2 %allows for a reasonable range
            useGCD = true;
            if unevenGCD < rescaleFactorBase 
                rescaleFactor = unevenGCD;
            else
                factorPossibilities = 1:unevenGCD;
                factorList = factorPossibilities(rem(unevenGCD,factorPossibilities)==0);
                sizePossibilities = unevenWidth./factorList;
                sizeDiscrepancies = abs(sizePossibilities-baseRescaleSize);
                [~,sizeIndex] = min(sizeDiscrepancies);
                rescaleFactor = 1/factorList(sizeIndex);
            end
        else
            useGCD = false;
            rescaleFactor = 1/floor(unevenWidth/baseRescaleSize);
        end
        unevenIlluminationSmall = imresize(unevenIllumination,rescaleFactor);

        %Fitting 2D Gaussian to uneven illumination image
        [smallHeight,smallWidth] = size(unevenIlluminationSmall);
        heightVector = 1:unevenHeight;
        widthVector = 1:unevenWidth;
        a0 = unevenMax; %max value 
        h0 = smallHeight/2;
        w0 = smallWidth/2; %center of image
        sigmaH0 = smallHeight/2;
        sigmaW0 = smallWidth/2; %guess
        x0 = double([a0,h0,w0,sigmaH0,sigmaW0]);
        f = @(x)GaussFit2D(x,unevenIlluminationSmall); %anonymous function
        options = optimoptions(@lsqnonlin,'Algorithm','levenberg-marquardt');
        xFinal = lsqnonlin(f,x0,[],[],options);
        
        %Plotting Gaussian fit
        A = xFinal(1);
        h0 = xFinal(2);
        w0 = xFinal(3);
        sigmaH = xFinal(4);
        sigmaW = xFinal(end); %4 if only 4 components, in which case both sigmas always the same

        gaussImageSmall = zeros(smallHeight,smallWidth);
        for h = 1:smallHeight
            for w = 1:smallWidth
                gaussExponent = -(h-h0)^2/(2*sigmaH^2)-(w-w0)^2/(2*sigmaW^2);
                gaussImageSmall(h,w) = A*exp(gaussExponent);
            end
        end
        
        %Translating Gaussian image back to larger one
        gaussImage = imresize(gaussImageSmall,1/rescaleFactor);
        %Fixing size rounding errors in height (imresize takes the ceiling,
        %so it will always be too big, not too small)
        if ~useGCD
            gaussImage = gaussImage(1:unevenHeight,:);
        end
        
        figure
        surf(widthVector,heightVector,gaussImage)
        hold on
        mesh(widthVector,heightVector,unevenIllumination)
        hold off
        zlabel('Image intensity');
        title('2D Gaussian fit to uneven illumination image');
        set(gca,'FontSize',15);
        saveas(gcf,[dirInput '/UnevenIlluminationGaussFit.png'])
        saveas(gcf,[dirInput '/UnevenIlluminationGaussFit.fig'])
        %Cutting off bits that are too low in intensity
        gaussImageRearranged = reshape(gaussImage,[unevenHeight*unevenWidth,1]);
        [gaussCDF,xValues] = ecdf(gaussImageRearranged);

        %Filtering image by combination of Gaussian and original image
        thresholdIndex = find(gaussCDF>unevenFraction,1);
        threshold = xValues(thresholdIndex);
        imageMask = ones(unevenHeight,unevenWidth);
        imageMask(gaussImage<threshold)=0; %Gaussian
        imageMask(unevenIllumination<threshold)=0; %Original image (some of these will be put back later)

        %Take median blur and restore removed regions
        imageMask = medfilt2(imageMask,[3 3]); %removes any little speckly bits
        imageMask = bwconvhull(imageMask); %gets rid of any random dark spots in the middle
        figure
        imshow(imageMask,[])
        title('Uneven Illumination-based Mask');
        imwrite(imageMask,[dirInput '/imageMask.tif']);
    else
        imageMask = ones(size(unevenIllumination)); %the identity mask; doesn't do anything
    end
end

%% Correcting images and saving corrected images

fileInformation = {};
%Identify subfolders and process images within them.
allFolders = allFiles;
allFolders(~[allFolders.isdir]) = []; %removes files that aren't folders
dirPath = [dirInput '/'];
for i = 1:numel(allFolders)
    currentFolder = allFolders(i).name;
    currentPath = [dirPath currentFolder];
    currentPath = [currentPath '/'];
    filesInFolder = dir(currentPath);
    for j = 1:numel(filesInFolder)
        fileName = filesInFolder(j).name;
        if numel(fileName) >= 4 && ~contains(fileName,'CellSegmentation.tif')  %used to remove things with filenames too short to have .tif
            fileExtension = fileName(end-3:end);
            shortFileName = fileName(1:end-4); %everything except the extension
            if (strcmp(fileExtension,'.tif') || strcmp(fileExtension,'tiff')) %if this is true, correct the image
                if strcmp(fileExtension,'tiff') %corrects the fact that I assumed the file name would be shorter
                    fileExtension = '.tiff';
                    shortFileName = fileName(1:end-5);
                end
                fprintf('Correcting %s... \n',fileName);
                [ImgInput, height, width, numFrames, tifData] = loadTiffStack(fileName,currentPath,zSlices);
                if zSlices == 0
                    try
                        zSlices = tifData.ImageDepth;
                        if zSlices == 1
                            warning('This image was not a hyperstack; Z-slices set to 1.')
                        end
                    catch
                        warning('This image did not have data on the number of Z-slices, setting to 1.')
                        zSlices = 1;
                    end
                end
                
                %correcting image for camera noise and uneven illumination
                if correctNoiseUneven
                    for z = 1:zSlices
                        for t = 1:numFrames
                            imageMaskNaN = NaN(height,width);
                            imageMaskNaN(imageMask==1)=1;
                            ImgInput(:,:,z,t) = ImgInput(:,:,z,t) - cameraNoise;
                            ImgInput(:,:,z,t) = ImgInput(:,:,z,t) ./ unevenIllumination;
                            ImgInput(:,:,z,t) = ImgInput(:,:,z,t) .* imageMaskNaN; %mask derived from uneven illumination image
                        end
                    end
                end
                
                %Finding sum and correcting for photobleaching
                
                %Creating z-projection sum

                ImgSum4D = sum(ImgInput,3);
                ImgSum = 0;
                clear ImgSum
                ImgSum(:,:,:) = ImgSum4D(:,:,1,:);
                
                %Segmenting cells
                
                %Checking for existing segmentation image
                if (exist([currentPath shortFileName '_CellSegmentation.tif'],'file') && newSegmentation==false) %if a segmented image exists and should be used
                    %Use existing segmentation
                    [cellTemplate,~,~,numFramesToSegment] = loadTiffStack([shortFileName '_CellSegmentation.tif'],currentPath,1,1); %Image format: x,y,t
                    cellTemplateMax = max(max(max(cellTemplate))); %in case externally created in ImageJ, then often has max 255
                    cellTemplatesFinal = NaN(height,width,numFramesToSegment);
                    for m = 1:numFramesToSegment
                        cellTemplateFinal = NaN(height,width);
                        cellTemplateFinal(cellTemplate==cellTemplateMax) = 1;
                        cellTemplatesFinal(:,:,m) = cellTemplateFinal;
                    end
                    segmentAutomatically = false;
                    frameDistance = ceil(numFrames/doSegmentationTimes);
                    if frameDistance < numFramesToSegment %this would cause the last segmentation to fail
                        numFramesToSegment = frameDistance;
                    end
                    framesArray = 1:frameDistance:numFrames;
                else  
                    segmentAutomatically = false;
                    if numFramesToSegment > numFrames 
                        numFramesToSegment = numFrames;
                    end
                    if (doSegmentationTimes > numFrames - numFramesToSegment + 1)
                        doSegmentationTimes = numFrames - numFramesToSegment + 1;
                    end
                    frameDistance = ceil(numFrames/doSegmentationTimes);
                    if frameDistance < numFramesToSegment %this would cause the last segmentation to fail
                        numFramesToSegment = frameDistance;
                    end
                    framesArray = 1:frameDistance:numFrames;
                    if automaticSegmentation
                        segmentAutomatically = true;
                    else
                        response = questdlg(['How should ' fileName ' be segmented?'],['Segmentation of ' fileName],'Automatically','Manually','No segmentation','Automatically');
                        if strcmp(response,'Automatically')
                            segmentAutomatically = true;
                        elseif strcmp(response,'Manually')
                            figure
                            cellTemplatesFinal = NaN(height,width,numFramesToSegment);
                            for m = 1:doSegmentationTimes 
                                imshow(ImgSum(:,:,framesArray(m)),[]);
                                title('Draw desired region. Double click when finished')
                                cellSegmentation = roipoly;
                                cellTemplateFinal = NaN(height,width);
                                cellTemplateFinal(cellSegmentation>0) = 1;
                                cellTemplatesFinal(:,:,m) = cellTemplateFinal;
                                if m == 1
                                    imwrite(cellTemplateFinal,[currentPath shortFileName '_CellSegmentation.tif']); %creates the tiff file
                                else
                                    imwrite(cellTemplateFinal,[currentPath shortFileName '_CellSegmentation.tif'],'WriteMode','append'); %adds a new image to the existing tiff file
                                end
                            end
                        else %no segmentation
                            cellTemplatesFinal = ones(height,width,numFramesToSegment);
                        end
                    end
                end
                
                if segmentAutomatically
                    cellTemplatesFinal = NaN(height,width,numFramesToSegment);
                    for m = 1:doSegmentationTimes %FIX THIS
                        cellTemplate = zeros(height,width);
                        startFrameNumber = framesArray(m);
                        for frameNumber = 1:numFramesToSegment
                            binSize = segmentationBinSize;
                            %turn first image into histogram
                            ImgHist = zeros(height,width);
                            for x = 1:height
                                for y = 1:width
                                    ImgHist(x,y) = floor(ImgSum(x,y,startFrameNumber+frameNumber-1)/binSize);
                                end
                            end
                            minBin = min(min(ImgHist));
                            ImgHist = imadd(ImgHist,-minBin+1); %sets minimum value of image histogram to 1
                            maxBin = max(max(ImgHist));
                            bins = zeros(maxBin,1);
                            for x = 1:height
                                for y = 1:width
                                    currentBin = ImgHist(x,y);
                                    if ~isnan(currentBin)
                                        bins(currentBin) = bins(currentBin) + 1;
                                    end
                                end
                            end

                            %Fitting the initial peak with two types of Gaussians and finding the
                            %cutoff between pixels in background and pixels in cells

                            %Part 1: finding the maximum
                            [~,peaklocs,~,peakIntensities] = findpeaks(bins);
                            [~,highestPeak] = max(peakIntensities);

                            %Part 2: Isolating the peak
                            backgroundxAxis = 1:peaklocs(highestPeak)*2;
                            backgroundBins = bins(1:peaklocs(highestPeak)*2);
                            backgroundxAxis = flip(rot90(backgroundxAxis));
                            fGauss = fit(backgroundxAxis(1:peaklocs(highestPeak)),backgroundBins(1:peaklocs(highestPeak)),'gauss1');
                            fGauss2 = fit(backgroundxAxis,backgroundBins,'gauss2');

                            %Part 3: Finding the fraction of pixels that are part of background as a
                            %function of ImgHist value
                            %Basically, pixels in the background go inside the first Gaussian, pixels
                            %in the cell go in the second Gaussian but not the first one
                            fractionsInCell = zeros(peaklocs(highestPeak)+1,1);
                            for k = 1:peaklocs(highestPeak)+1
                                x = k + peaklocs(highestPeak)-1;
                                fractionsInCell(k) = (fGauss2(x) - fGauss(x))/fGauss2(x);
                            end

                            minArray = find(fractionsInCell>threshold);
                            if numel(minArray) > 0
                                minimum = minArray(1) + peaklocs(highestPeak) - 1;
                            else
                                minimum = peaklocs(highestPeak)*2;
                            end

                            %Plotting everything
                             if(frameNumber==1 && plotAllGraphs)
                                figure
                                hold on
                                area(backgroundxAxis,fGauss2(backgroundxAxis),'FaceColor',[1 0.75 0.75]);
                                area(backgroundxAxis,fGauss(backgroundxAxis),'FaceColor',[0.75 0.75 1]);
                                plot(backgroundxAxis,backgroundBins,'o','MarkerSize',7,'color','black')
                                x = [minimum minimum];
                                y = [0 max(backgroundBins)*1.1];
                                line(x,y,'Color','black','LineStyle','--')
                                hold off

                                xlabel('Intensity bin (AU)')
                                ylabel('Number of pixels')
                                xlim([0 backgroundxAxis(end)])
                                ylim([0 max(backgroundBins)*1.1]);
                                legend('Intracellular pixels','Background pixels','Raw data')
                                title('Gaussian fits to pixel brightness histogram')
                                set(gca,'FontSize',20)
                                set(legend,'FontSize',20)
                                saveas(gcf,[currentPath shortFileName '_PixelBrightnessHistogram_' num2str(m) '.fig'])
                             end
                            cellTemplateCurrent = zeros(height,width);
                            cellTemplateCurrent(ImgHist(:,:)>=minimum) = 1;
                            cellTemplate = cellTemplate + cellTemplateCurrent;
                        end
                        binaryImage = zeros(height,width);
                        numFramesThreshold = numFramesToSegment*fractionToPass;
                        binaryImage(cellTemplate>=numFramesThreshold)=1;
                        binaryImage = medfilt2(binaryImage); %cleans up noise slightly; probably not really necessary
                        cellTemplateFinal = NaN(height,width);
                        cellTemplateFinal(binaryImage==1) = 1;
                        cellArea = sum(sum(binaryImage));
                        %figure
                        %imshow(cellTemplateFinal);
                        %title([shortFileName ' Image Segmentation'],'FontSize',15);
                        %saveas(gcf,[currentPath shortFileName '_CellSegmentation.png'])
                        %imwrite(cellTemplateFinal,[currentPath shortFileName '_CellSegmentation.tif']);
                        figure
                        imshow(cellTemplateFinal,[]);
                        if m == 1
                            imwrite(cellTemplateFinal,[currentPath shortFileName '_CellSegmentation.tif']); %creates the tiff file
                        else
                            imwrite(cellTemplateFinal,[currentPath shortFileName '_CellSegmentation.tif'],'WriteMode','append'); %adds a new image to the existing tiff file
                        end

                        cellTemplatesFinal(:,:,m) = cellTemplateFinal;
                    end
                end
                
                %Finding median intracellular brightness in each frame
                medianBrightness = zeros(numFrames,1);
                for k = 1:numFrames
                    curSegmentationFrame = floor((k/(numFrames+1))*doSegmentationTimes)+1; %figures out which segmentation to use
                    %find median of all pixels greater than minimum
                    ImgVector = zeros(height*width,1); %convert image into 1D vector so you can get the median
                    ImgCells = ImgSum(:,:,k);
                    ImgCells = ImgCells .* cellTemplatesFinal(:,:,curSegmentationFrame); %makes all the non-cell entries NaN, leaves others unaltered
                    for x = 1:height
                        for y = 1:width
                            currentElement = (x-1)*(height)+y;
                            ImgVector(currentElement) = ImgCells(x,y);
                        end
                    end
                    medianBrightness(k) = median(ImgVector,'omitnan');
                end

                %plotting and fitting the curve
                xAxis = 1:numFrames;
                xAxis = flip(rot90(xAxis));
                if strcmp(fitMethod,'none') %'none' just uses the raw data as the "curve"
                    photoBleachingCurve = medianBrightness;
                    figure;
                    plot(photoBleachingCurve);
                    xlabel('Time (timepoints)');
                    ylabel('Median brightness in cells (AU)');
                    hold on;
                    set(gca,'FontSize',15)
                    saveas(gcf,[currentPath shortFileName '_PhotoBleachingCurve.fig']);
                    saveas(gcf,[currentPath shortFileName '_PhotoBleachingCurve.png']);
                else
                    [f,GoF] = fit(xAxis,medianBrightness,fitMethod); 
                    fitCoefficients = coeffvalues(f);
                    xMax = numFrames;
                    yMax = max(medianBrightness);
                    figure;
                    photoBleachingCurve = f(xAxis);
                    plot(f,xAxis,medianBrightness,'+b');
                    hold on
                    %plot(xAxis,photoBleachingCurve,'-r');
                    hold off
                    xlabel('Time (timepoints)');
                    ylabel('Median brightness in cells (AU)');
                    ylim([0 yMax]);
                    xlim([1 xMax]);
                    title([shortFileName ' Photobleaching Curve'])
                    set(gca,'FontSize',15)
                    %Generating and adding equations
                    if strcmp(fitMethod,'exp2')
                        fitFormat = 'y=$(%.1f)e^{%.2fx}+(%.1f)e^{%.2fx}$'; 
                        fitText = sprintf(fitFormat,fitCoefficients);
                        text(xMax,0.9*yMax,fitText,'interpreter','latex','HorizontalAlignment','right','FontSize',15)
                    elseif strcmp(fitMethod,'exp1')
                        fitFormat = 'y=$(%.1f)e^%.2fx$';
                        fitText = sprintf(fitFormat,fitCoefficients);
                        text(xMax,0.9*yMax,fitText,'interpreter','latex','HorizontalAlignment','right','FontSize',15)
                    end
                    l = legend('placeholder text'); %sorry for bad code
                    set(l,'visible','off')
                    saveas(gcf,[currentPath shortFileName '_PhotoBleachingCurve.fig']);
                    saveas(gcf,[currentPath shortFileName '_PhotoBleachingCurve.png']);
                end 
                
                if saveCorrectedImages
                    %generating the photobleaching-corrected stack
                    photoBleachingCurve = photoBleachingCurve/(photoBleachingCurve(1)); %normalization
                    for z = 1:zSlices
                        for t = 1:numFrames
                            ImgInput(:,:,z,t) = ImgInput(:,:,z,t)/photoBleachingCurve(t);
                        end
                    end

                    %Creating corrected image
                    warning('off','all')
                    if overWrite
                        copyfile([currentPath fileName],[currentPath shortFileName '_Uncorrected.tif']);
                        correctedFile = Tiff([currentPath fileName],'r+');
                    else
                        copyfile([currentPath fileName],[currentPath shortFileName '_Corrected.tif']);
                        correctedFile = Tiff([currentPath shortFileName '_Corrected.tif'],'r+');
                    end

                    %Determining data type and bits
                    bitString = num2str(tifData.BitsPerSample);
                    if tifData.SampleFormat == 3 %IEEEFP format; floating point
                        dataType = 'single'; %BitsPerSample should be 32, I believe
                    elseif tifData.SampleFormat == 2 %Signed integer
                        dataType = ['int' bitString];
                    elseif tifData.SampleFormat == 1 %Unsigned integer
                        dataType = ['uint' bitString];
                    else
                        error('Image type not recognized. Could not save') %this shouldn't actually happen
                    end
                    for t = 1:numFrames 
                        for z = 1:zSlices
                            ImgFinalCurrent = cast(ImgInput(:,:,z,t),dataType); %converts to correct data type
                            correctedFile.setDirectory((t-1)*zSlices+z);
                            correctedFile.write(ImgFinalCurrent,[currentPath fileName]);
                        end
                    end        
                    correctedFile.close(); 
                end
            end
        end
    end
end

%% Creating fake images
if createFakeImages
    fprintf('Creating fake images for PatchTrackingTools... \n')
    fakeNoise = zeros(height,width);
    imwrite(fakeNoise,[dirPath 'AverageCameraNoise.tif']);
    fakeIllumination = uint16(ones(height,width)); %uint16 because otherwise when it saves the image all values are 255
    imwrite(fakeIllumination,[dirPath 'AverageUnevenIllumination.tif']);
end
fprintf('All done! \n');


