function [x,iter]=D_ADMM_H( y,H,miu,method, eps)
%D_ADMM A Derivative-Space alternating directional method of mutipliers for TV-based image restoration
%    The method is designed based on the ALM
%    The constraint d=Dx requires d lies in the irrotatioanl subspace V.
%    According to the definition of curl of a vector, we hereby rewrite the constraint as
%    a linear form, i.e., Dh{dv}=Dv{dh}
%
%    Input:
%         y:      degraded image
%         H:      linear oprater
%         miu:    weighting parameter, controlling a trade-off between data
%                 fidelity and TV item.
%         method: options for selecting TV operators, i.e., 1, the default
%                 setting, for anisotropic version and others for isotropic
%                 version.
%         eps:    stopping criterion, default is 1e-4.
%
%    Output:
%         x:      reconstructed image
%         iter:   iteration numbers of the restoration procedure
%

%%
if nargin<5
    eps=1e-4;
end

%% preprocessing
[m,n]=size(y);
mean_y=sum(y(:))/(m*n);

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
Wi=1./(2*cos(2*pi*I/m)+2*cos(2*pi*J/n)-4);Wi(1)=0;

Dh_FFT=1-exp(-1i*2*pi*J/n);% backward gradient operator, with d(0)=x(0)-x(N-1)
Dv_FFT=1-exp(-1i*2*pi*I/m);
cDh_FFT=conj(Dh_FFT);
cDv_FFT=conj(Dv_FFT);

denominator=cDh_FFT.*Dh_FFT+cDv_FFT.*Dv_FFT;
denominator(1)=1;% in case of 0/0

%% initilation
[dx,dy]=BackwardD(y);
px=zeros(m,n);py=px;

%%
H_FFT=psf2otf(H,[m,n]);
HC_FFT = conj(H_FFT);

AsDhy=HC_FFT.*fft2(dx);%A^T{D_h{y}}
AsDvy=HC_FFT.*fft2(dy);%A^T{D_v{y}}

ATA_FFT=H_FFT.*HC_FFT;

%% parameters setting
delta=1e-1;
% delta_max=100;
maxIter=200;

% psnr_t=[];
% ssim_t=[];
% 
% psnr_t=[psnr_t;psnr(x_true,y)];
% ssim_t=[ssim_t;ssim(x_true*255,y*255)];

%%
for i=1:maxIter
    
    %% f subproblem
    if method==1
        [fx,fy] = anisoTVDenoise(miu/delta,dx+px,dy+py);
    elseif method==2
        [fx,fy] = isoTVDenoise(miu/delta,dx+px,dy+py);
    else
         fx=solve_Lp(dx+px,miu/delta,.8);
         fy=solve_Lp(dy+py,miu/delta,.8);
    end
    
    %% d subproblem
    dxp=dx;dyp=dy;%store the results of previous iteration
    
    fpx=delta*fft2(fx-px);
    fpy=delta*fft2(fy-py);
    
    lamb=(cDv_FFT.*(AsDhy+fpx)-cDh_FFT.*(AsDvy+fpy))./denominator;% Lagrangian multiplier
    dx=real(ifft2((AsDhy+fpx-Dv_FFT.*lamb)./(ATA_FFT+delta)));
    dy=real(ifft2((AsDvy+fpy+Dh_FFT.*lamb)./(ATA_FFT+delta)));
    
    %%
%     x=real(ifft2(fft2(div(dx,dy)).*Wi));% x=U{dx,dy}
%    psnr_t=[psnr_t;psnr(x_true,x+mean_y)];
% ssim_t=[ssim_t;ssim(x_true*255,(x+mean_y)*255)];

    %% check the stopping criterion
    normdx=norm((dxp(:)-dx(:)));
    normdy=norm((dyp(:)-dy(:)));
    if normdx/norm(dxp(:))+ normdy/norm(dyp(:))<=eps
        iter=i;
        break;
    end
    
    %% update the parameters
    px=px+dx-fx;
    py=py+dy-fy;
    
    if (normdx+normdy)*delta/(norm(fx(:))+norm(fy(:))) < 1e-3
        rao=1.9;
    else
        rao=1;
    end
    
    delta=rao*delta;
    %     delta=min(delta_max,rao*delta);
    
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

    function DtXY = div(X,Y)
        % divergence of the backward finite difference operator
        DtXY = [ diff(X,1,2),X(:, 1)-X(:,end) ];
        DtXY = DtXY + [ diff(Y,1,1);Y(1, :)-Y(end,:)];
    end

end