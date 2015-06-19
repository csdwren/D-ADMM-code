function [x,iter]=D_ADMM_C(  y,H,miu,method, eps)
%D_ADMM A Derivative-Space alternating directional method of mutipliers for TV-based image restoration
%    The method is designed based on the ALM framework, i.e., introducing two constraints,
%    then adddressing several subproblems.
%    The constraint d=Dx requires d lies in the irrotatioanl subspace
%    V. According to the definition of curl of a vector, we hereby rewrite the constraint as
%    a linear form, i.e., Dh{dv}=Dv{dh}, and it is totally solved by ADMM.
%
%    Input:
%         y:      degraded image
%         H:      linear oprater
%         miu:    weighting parameter, controlling a trade-off between data
%                 fidelity and TV item.
%         varargin: {1} options for selecting TV operators, i.e., 1, the default
%                 setting, for anisotropic version and others for isotropic
%                 version.
%                   {2} eps, stopping criterion, default is 1e-3.
%                   {3} option for if compute objective function error.
%                   default:0, ~0 for yes.
%    Output:
%         x:      reconstructed image
%         iter:   iteration numbers of the restoration procedure
%      Optional output arguments,
%         Ferr:   objective function values of each iteration.
%

%%
%% preprocessing
[m,n]=size(y);
mean_y=sum(y(:))/(m*n);
% y=y-mean_y; %unnecessary processing

%%
J=(0:(n-1));J=repmat(J,[m,1]);
I=(0:(m-1))';I=repmat(I,[1,n]);
% I=zeros(m,n);J=I;
% for k=1:m
%     J(k,:)=(0:(n-1));
% end
% for k=1:n
%     I(:,k)=(0:(m-1));
% end
Wi=1./(2*cos(2*pi*I/m)+2*cos(2*pi*J/n)-4);Wi(1,1)=0;

Dh_FFT=1-exp(-1i*2*pi*J/n);% backward gradient operator, with d(0)=x(0)-x(N-1)
Dv_FFT=1-exp(-1i*2*pi*I/m);

Dh2=Dh_FFT.*conj(Dh_FFT);
Dv2=Dv_FFT.*conj(Dv_FFT);

%% initilation
[dx,dy]=BackwardD(y);

p=zeros(m,n);
qx=p;qy=p;

%%
H_FFT=psf2otf(H,[m,n]);
HC_FFT = conj(H_FFT);

AsDhy=HC_FFT.*fft2(dx);%A^T{D_h{y}}
AsDvy=HC_FFT.*fft2(dy);%A^T{D_v{y}}

ATA_FFT=H_FFT.*HC_FFT;

%% parameters setting
delta2=1e-4;
delta1=delta2;
delta_max=100*delta1;
maxIter=200;

% psnr_t=[];
% ssim_t=[];
% 
% psnr_t=[psnr_t;psnr(x_true,y)];
% ssim_t=[ssim_t;ssim(x_true*255,y*255)];

%%
for i=1:maxIter
    
    %% f
    if method==1
        [fx,fy] = anisoTVDenoise(miu/delta2,dx+qx,dy+qy);
    else
        [fx,fy] = isoTVDenoise(miu/delta2,dx+qx,dy+qy);
    end
    
    %% d
    dxp=dx;dyp=dy;%store the results of previous iteration
    
    tmp=BackwardDy(BackwardDxT(dy)+p);
    dx=real(ifft2((AsDhy+fft2(delta1*tmp+delta2*(fx-qx)))./(ATA_FFT+delta1*Dv2+delta2)));
    tmp=BackwardDx(BackwardDyT(dx)-p);
    dy=real(ifft2((AsDvy+fft2(delta1*tmp+delta2*(fy-qy)))./(ATA_FFT+delta1*Dh2+delta2)));
    
    %%
%     x=real(ifft2(fft2(div(dx,dy)).*Wi));% x=U{dx,dy}
%    psnr_t=[psnr_t;psnr(x_true,x+mean_y)];
% ssim_t=[ssim_t;ssim(x_true*255,(x+mean_y)*255)];

    
    %     x=real(ifft2(fft2(div(dx,dy)).*Wi));
%         imshow(x+mean_y);pause();
    
    %% check the stopping criterion
    dxd=dxp-dx;
    dyd=dyp-dy;
    
    normdx=norm(dxd(:));
    normdy=norm(dyd(:));
    if normdx/norm(dxp(:))+ normdy/norm(dyp(:))<=4*eps
        iter=i;
        break;
    end
    
    %% update the parameters
    p=p+BackwardDxT(dy)-BackwardDyT(dx);
    
    qx=qx+dx-fx;
    qy=qy+dy-fy;
    
    a=BackwardDxT(dyd);b=BackwardDyT(dx);
    if norm(a(:))*delta1/(norm(b(:)))<1e-3
        rao1=1.5;
    else
        rao1=1;
    end
    
    if (normdx+normdy)*delta2/(norm(fx(:))+norm(fy(:)))<1e-3
        rao2=2.9;
    else
        rao2=1;
    end
    
    delta1=min(rao1*delta1,delta_max);
    delta2=min(rao2*delta2,delta_max);
    
end
%%
if i==maxIter
    iter=maxIter;
end

%% obtain recovered image according to its gradient
x=real(ifft2(fft2(div(dx,dy)).*Wi));% x=U{dx,dy}
x=x+mean_y;

%% nested functions
    function [Dux,Duy] = BackwardD(U)
        % Backward finite difference operator
        Dux = [ U(:,1) - U(:,end),diff(U,1,2)];
        Duy = [ U(1,:) - U(end,:);diff(U,1,1)];
    end

    function [Dux,Duy] = BackwardDT(U)
        % Backward finite difference operator
        Dux = [ -diff(U,1,2), U(:,end)-U(:,1)];
        Duy = [ -diff(U,1,1); U(end,:)-U(1,:)];
    end

    function Dux=BackwardDx(U)
        Dux = [ U(:,1) - U(:,end),diff(U,1,2)];
    end
    function Duy=BackwardDy(U)
        Duy = [ U(1,:) - U(end,:);diff(U,1,1)];
    end

    function Dux = BackwardDxT(U)
        % Backward finite difference operator
        Dux = [ -diff(U,1,2), U(:,end)-U(:,1)];
    end
    function Duy = BackwardDyT(U)
        % Backward finite difference operator
        Duy = [ -diff(U,1,1); U(end,:)-U(1,:)];
    end

    function DtXY = div(X,Y)
        % divergence of the backward finite difference operator
        DtXY = [ diff(X,1,2),X(:, 1)-X(:,end) ];
        DtXY = DtXY + [ diff(Y,1,1);Y(1, :)-Y(end,:)];
    end

end

