function R = MTMLd_3(XR, V)
% R = (XR'*V')*inv(V*V');
R = (XR'*V')/(V*V');
end