function V = MTMLd_2(R,U,XR,Z,Gamma, rho1,tau)
% V = inv(rho1*R'*R+ 2*tau*(U'*U))*(rho1*R'*XR'+U'*Gamma+2*tau*U'*Z);
V = (rho1*(R'*R)+ 2*tau*(U'*U))\(rho1*R'*XR'+U'*Gamma+2*tau*U'*Z);
end