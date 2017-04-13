dataset = imageSet('/Users/bruno/Desktop/EmotionFacesDataAneesh/', 'recursive');
dataset = imageSet('/Users/bruno/Desktop/EmotionFacesDataAneeshLess/', 'recursive');

lbpFeatures = [];
labels = [];
allFeatures = [];
hogFeatures = [];

[train, test] = partition(dataset, [0.8, 0.2]);

faceDetector = vision.CascadeObjectDetector;
NoseDetect = vision.CascadeObjectDetector('Nose','MergeThreshold',16);
MouthDetect = vision.CascadeObjectDetector('Mouth','MergeThreshold',16);
EyeDetect = vision.CascadeObjectDetector('EyePairBig');

totalImg = 0;
counter = 0;
for i = 1:size(train,2) %get how many classes i have
  for j = 1:train(i).Count
      
      I = read(train(i),j);  
      
      if size(I, 3) == 3
          I = rgb2gray(I);
          counter = counter + 1;
      end
      
      bbox = step(faceDetector, I);
      
      onlyFace = imcrop(I,bbox(1,1:4));
      onlyFace = imresize(onlyFace, [300 300]);
      onlyMouth = imcrop(onlyFace, [45.5100 210.5100 222.9800 78.9800]);
      
      %boxEye = step(EyeDetect, onlyFace);
      %boxMouth = step(MouthDetect,onlyFace);
      %boxNose = step(NoseDetect,onlyFace);
      
      %nose = imcrop(onlyFace, boxNose);
      %mouth = imcrop(onlyFace, boxMouth(1,1:4));
      %eyes = imcrop(onlyFace, boxEye);
      
      featLbp = extractLBPFeatures(onlyFace);
      lbpFeatures = double(vertcat(lbpFeatures, featLbp));
      
      %[hogFeat,validPoints] = extractHOGFeatures(onlyFace);
      %hogFeatures = double(vertcat(hogFeatures, hogFeat));
      
      labels = vertcat(labels, {train(i).Description});
      
      %allFeatures = horzcat(lbpFeatures, hogFeatures);
      allFeatures = horzcat(lbpFeatures);
      
      totalImg = totalImg + 1;
  end
end

FacesTraining = array2table(allFeatures);
FacesTraining.name = labels;

disp('')