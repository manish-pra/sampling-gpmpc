clear all
clc
close all
a=0.9;b=0.2;%parameter, linear system: x^+=ax+bu+w
T=30;t=0:T-1;%time period for samples
u=sin(t*2*pi/T);%simple input excitation
M=1e3; %number of experiments to check if biased
w=normrnd(0,1,T,M);%sample Gaussian noise
x_0=0;x(1,1:M)=x_0;%initial condition
for i=1:M
    %different realizations    
   for j=1:T
      %simulate
      x(j+1,i)=a*x(j,i)+b*u(j)+w(j,i);   
   end
   %identify parameters (a,b) using least-squares
   Phi=[x(1:T,i)';u(1:T)];
   Y=x(2:T+1,i);
   P=inv(Phi*Phi');
   hat_theta(:,i)=P*Phi*Y;
end 
disp('compare empirical mean to true parameter')
mean(hat_theta')'
[a;b] 