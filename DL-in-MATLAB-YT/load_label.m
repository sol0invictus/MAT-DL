function Y = load_label(filename)
% The processLabelsMNIST function operates similarly to the
% processImagesMNIST function. After opening the file and reading the magic
% number, it reads the labels and returns a categorical array containing
% their values.

fileID = fopen(filename,'r','b');

magic = fread(fileID,1,'int32',0,'b');
if magic == 2049
    disp('Reading MNIST Labels.......')
end

num_labels = fread(fileID,1,'int32',0,'b');
Y = fread(fileID,inf,'unsigned char');
disp('done loading');
fclose(fileID);
end