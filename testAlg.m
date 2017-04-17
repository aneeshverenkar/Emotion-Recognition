dataset = imageSet('/Users/bruno/Desktop/EmotionFacesDataAneesh/', 'recursive');
%dataset = imageSet('/Users/bruno/Desktop/EmotionFacesDataAneeshLess/', 'recursive');

lbpFeatures = [];
labels = [];
allFeatures = [];
hogFeatures = [];

[train, test] = partition(dataset, [0.8, 0.2]);

%viola jones
faceDetector = vision.CascadeObjectDetector;
NoseDetect = vision.CascadeObjectDetector('Nose','MergeThreshold',16);
MouthDetect = vision.CascadeObjectDetector('Mouth','MergeThreshold',16);
EyeDetect = vision.CascadeObjectDetector('EyePairBig');

totalImg = 0;
counter = 0;
for i = 1:size(train,2) %get how many classes i have
  for j = 1:train(i).Count
      
      I = read(train(i),j);  
      I = imresize(I, [300 300]);
      
      %verify if image has color
      if size(I, 3) == 3
          I = rgb2gray(I);
      end
      
      WIDTH = 3;
      HEIGHT = 4;
      index = 1;
      indexEye = 1;
      counter = 0;
      
      bbox = step(faceDetector, I);
      
      %gets correct face
      if size(bbox) > 1
          for bbIndex = 1 : size(bbox, 1)
              if bbox(bbIndex,WIDTH) > 140 && bbox(bbIndex,HEIGHT) > 140 %[x y w h]
                  index = bbIndex;
              end
          end
          counter = counter + 1;
      end
      
      onlyFace = imcrop(I,bbox(index, 1:4));
      onlyFace = imresize(onlyFace, [300 300]);
      boxEye = step(EyeDetect, onlyFace);
      
      %gets correct eye
      if size(boxEye) > 1
         for boxEyeIndex = 1 : size(boxEye, 1)
             if boxEye(boxEyeIndex,WIDTH) > 140 && boxEye(boxEyeIndex,HEIGHT) < 60 %[x y w h]
                 indexEye = boxEyeIndex;
             end
         end
         counter = counter + 1;
      end
      
      onlyEyes = imcrop(onlyFace, [48.5100 73.5100 197.9800 66.9800]);
      onlyMouth = imcrop(onlyFace, [61.5100 218.5100 174.9800 66.9800]);
      
      
      %extracts lbp
      mouthLbp = extractLBPFeatures(onlyMouth);
      eyesLbp = extractLBPFeatures(onlyEyes);
      
      %extracts hog
      mouthHog = extractHOGFeatures(onlyMouth);
      eyesHog = extractHOGFeatures(onlyEyes);
      
      %concatenate hog and lbp features
      concatHogFeat = horzcat(mouthHog, eyesHog);
      concatLbpFeat = horzcat(mouthLbp, eyesLbp);
      
      %put hog and lbp features in the vector
      hogFeatures = double(vertcat(hogFeatures, concatHogFeat));
      lbpFeatures = double(vertcat(lbpFeatures, concatLbpFeat));
      
      labels = vertcat(labels, {train(i).Description});
      
      %put all features extracted in the main matrix
      allFeatures = horzcat(lbpFeatures, hogFeatures);
      
      totalImg = totalImg + 1;
  end
end

FacesTraining = array2table(allFeatures);
FacesTraining.name = labels;