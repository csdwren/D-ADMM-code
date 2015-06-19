function [ u,v ] = anisoTVDenoise( lambda,wh,wv ,varargin)
%    this function compute ||u,v||_l+lambda||u-wh||^2+||v-wv||^2
if nargin==4
    if size(wh)==size(wv)
        V=abs(wh);
        V(V==0)=1;
        V=max(V-lambda,0)./V;
        u=(wh/varargin{1}).*V;
        
        V=abs(wv);
        V(V==0)=1;
        V=max(V-lambda,0)./V;
        v=(wv/varargin{1}).*V;
        
    else
        u=zeros(size(wh));
        v=zeros(size(wv));
    end
elseif nargin==3
    if size(wh)==size(wv)
        V=abs(wh);
        V(V==0)=1;
        V=max(V-lambda,0)./V;
        u=(wh).*V;
        
        V=abs(wv);
        V(V==0)=1;
        V=max(V-lambda,0)./V;
        v=(wv).*V;
        
    else
        u=zeros(size(wh));
        v=zeros(size(wv));
    end
else
    error('Invilid Inputs!');
end
end

