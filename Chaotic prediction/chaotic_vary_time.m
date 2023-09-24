clc;
close all;
clear all;
Vmax=2;
Vmin=0;
datasize=2000;%Henon datasize
[x,y]=Henon(datasize+1);%generate Henon map
ratio=0.5;%train/datasize
n=25;%number of mask
m=50;%length of mask
%Henon_train
input_train=x(1:round(ratio*datasize));
target_train=x(2:round(ratio*datasize)+1);
%Henon_test
input_test=x(round(ratio*datasize)+1:datasize);
target_test=x(round(ratio*datasize)+2:datasize+1);
ntrain=length(input_train);%train datasize
ntest=length(input_test);%test datasize
mask=2*randi(2,n,m)-3;%generate mask
%train process
%mask process
train_mask=[];
for j=1:n
    for i=1:ntrain
        train_mask(j,(i-1)*m+1:m*i)=input_train(1,i)*mask(j,:);
    end
end
train_max=max(max(train_mask));
train_min=min(min(train_mask));
%voltage input
train_voltage=(train_mask-train_min)/(train_max-train_min)*(Vmax-Vmin)+Vmin;
%device output
for modu=0:50
    for relax=0:200
        current_output=device_sim_vary_time(train_voltage,relax,modu);
        %linear regression
        a=[];
        states=[];
        for i=1:ntrain
            a=current_output(:, m*(i-1)+1:m*i);
            states(:,i)=a(:);
        end
        input=[ones(1,ntrain);states];
        weight=target_train*pinv(input);
        %test process
        %mask process
        test_mask=[];
        for j=1:n
            for i=1:ntest
                test_mask(j,(i-1)*m+1:m*i)=input_test(1,i)*mask(j,:);
            end
        end
        test_max=max(max(test_mask));
        test_min=min(min(test_mask));
        %voltage input
        test_voltage=(test_mask-test_min)/(test_max-test_min)*(Vmax-Vmin)+Vmin;
        %device output
        current_output=device_sim_vary_time(test_voltage,relax,modu);
        %chaotic prediction
        a=[];
        states=[];
        for i=1:ntest
            a=current_output(:, m*(i-1)+1:m*i);
            states(:,i)=a(:);
        end
        input=[ones(1,ntest);states];
        output=weight*input;
        NRMSE(modu+1,relax+1)=sqrt(mean((output(10:end)-target_test(10:end)).^2)./var(target_test(10:end)));
    end
end