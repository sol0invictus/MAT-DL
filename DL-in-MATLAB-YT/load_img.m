function X = load_img(filename)

fileID = fopen(filename,'r','b');

magic = fread(fileID,1,'int32',0,'b');
if magic == 2051
    disp('reading MNIST.....')
end

num_images = fread(fileID,1,'int32',0,'b');
height = fread(fileID,1,'int32',0,'b');
width = fread(fileID,1,'int32',0,'b');

X = fread(fileID,inf,'unsigned char');
X = reshape(X,width,height,num_images);
X = permute(X,[2 1 3]);
X = X./255;
X = reshape(X,784,[]);
disp('done loading')
fclose(fileID);
end

