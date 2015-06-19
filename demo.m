
%%

x = imread('Barbara.jpg');
if(length(size(x))==3)
    x=im2double(rgb2gray(x));
else
    x=im2double(x);
end


sigma=5e-3;
miu=4e-4;


[m, n] = size(x);

%%
%%get the oberverd image
load kernels.mat
H=k{7};

H_FFT=psf2otf(H,[m,n]);
HC_FFT = conj(H_FFT);


y=imfilter(x,H,'circular','conv')+ sigma*randn(m,n);

tic;
% [x_admm,iter]=D_ADMM_C(y,H,miu,2,1e-4);
[x_admm,iter]=D_ADMM_H(y,H,miu,2,1e-4);
t=toc;

figure,imshow(x_admm);

