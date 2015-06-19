function [ u,v ] = isoTVDenoise( lambda,wh,wv ,varargin)
%    this function compute ||u,v||_l2+lambda||u-wh||^2+||v-wv||^2

if nargin==4
    if size(wh)==size(wv)
        V = wh.^2 + wv.^2;
        
        V = sqrt(V);
        V(V==0) = 1;
        V = max(V - lambda, 0)./V/varargin{1};
        u = wh.*V;
        v = wv.*V;
        
    else
        u=zeros(size(wh));
        v=zeros(size(wv));
    end
elseif nargin==3
    if size(wh)==size(wv)
        V = wh.^2 + wv.^2;
        
        V = sqrt(V);
        V(V==0) = 1;
        V = max(V - lambda, 0)./V;
        u = wh.*V;
        v = wv.*V;
        
    else
        u=zeros(size(wh));
        v=zeros(size(wv));
    end
else
    error('Invilid Inputs!');
end

end

