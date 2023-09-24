clc
clear
close all

load('group_binary.mat')
load('voice_bq.mat');
load('state.mat')
larger=100;
inputs=zeros(125,20*larger);

t=zeros(125,80,larger-1);
for i=1:larger-1
    t(:,:,i)=imnoise(group_binary(:,1:80),'salt & pepper',0.05);
    group_binary(:,80*i+1:80*i+80)=t(:,:,i);
end

% for i=1:20*larger %20个初始音频+80个带噪声音频
%     for j=1:125 
%         order=group_binary(j,4*i-3)*8+group_binary(j,4*i-2)*4+group_binary(j,4*i-1)*2+group_binary(j,4*i)+1;
%         inputs(j,i) =state(order);
%     end
% end

for i=1:20*larger %20个初始音频+80个带噪声音频
    for j=1:125 
        order=group_binary(j,4*i-3)+group_binary(j,4*i-2)+group_binary(j,4*i-1)+group_binary(j,4*i);
        inputs(j,i) =order  ;
    end
end

for i=1:larger-1
    bq(:,20*i+1:20*i+20)=bq(:,1:20);
end
inputs_max=max(max(inputs));
inputs_min=min(min(inputs));
inputs=(inputs-inputs_min)/(inputs_max-inputs_min);
train_inputs=inputs(:,1:10*larger);
train_bq=bq(:,1:10*larger);
test_inputs=inputs(:,10*larger+1:end);
test_bq=bq(:,10*larger+1:end);
[mtrain,ntrain]=size(train_bq);
[mtest,ntest]=size(test_bq);


arch = [125,5];
nlayer = length(arch);

mini_batch_size = 10*larger;% 数
max_epochs = 10000;% 
max_accu_epoch=10000;
zeta =0.0165;
threshold=0;
weight = cell(1,nlayer);
bias = cell(1,nlayer);

nabla_weight = cell(1,nlayer); 
nabla_bias = cell(1,nlayer);   

a = cell(1,nlayer);            
z = cell(1,nlayer);            

at = cell(1,nlayer);            
zt = cell(1,nlayer);            

for in = 2:nlayer
%     weight{in} = rand(arch(1,in),arch(1,in-1))*2-1;
%     bias{in} = rand(arch(1,in),1);
%     nabla_weight{in} = rand(arch(1,in),arch(1,in-1));
%     nabla_bias{in} = rand(arch(1,in),1);
    weight{in} =normrnd(0,0.33333,arch(1,in),arch(1,in-1));
    bias{in} = normrnd(0,0.33333,arch(1,in),1);
    nabla_weight{in} = normrnd(0,0.33333,arch(1,in),arch(1,in-1));
    nabla_bias{in} = normrnd(0,0.33333,arch(1,in),1);
end
weight0=weight{2};
for in = 1:nlayer
    a{in} = zeros(arch(in),mini_batch_size);
    z{in} = zeros(arch(in),mini_batch_size);
    at{in} = zeros(arch(in),ntest);
    zt{in} = zeros(arch(in),ntest);
end
% bp
time=0:max_epochs;
time=time';

accuracy=zeros(max_epochs+1,1);
%while(accuracy(max_epochs+1,1)~=100)

s=0;

at{1}=test_inputs;
for in = 2:nlayer
    wt = weight{in};
    bt = bias{in};
    ixt = at{in-1};
    izt = wt*ixt;
    for im = 1:ntest
        izt(:,im) = izt(:,im)+bt;
    end
    if in==nlayer
        zt{in} = izt;
        exat=exp(izt);
        stant=sum(exat);
        for im = 1:ntest
            at{in}(:,im)=exat(:,im)/stant(1,im);
        end
    else
        zt{in} = izt;
        at{in} = sigmoid(izt);
    end
end
for m=1:ntest
    [dx,wz]=max(at{nlayer}(:,m));
    [dx2,wz2]=max(test_bq(:,m));
    if wz==wz2
        s=s+1;
    end
end
accuracy(1,1)=s/ntest;
fprintf('0  %.2f\n',accuracy(1,1)*100)

s=0;
CrossEntropy=zeros(1,max_epochs);
for ip = 1:max_epochs
%     if ip<1800
%        zeta=0.0163;
%     else
%         zeta=0.017;
%     end
%     random_series=randperm(20);
%     input = inputs(:,random_series(1:mini_batch_size));
%     output = bq(:,random_series(1:mini_batch_size));
    %input=inputs(:,1:20);
    %output=bq(:,1:20);
    input=train_inputs;
    output=train_bq;

    a{1} = input;
    if(accuracy(ip,1)~=1)
    for in = 2:nlayer
        w = weight{in};
        b = bias{in};
        ix = a{in-1};
        iz = w*ix;
        for im = 1:mini_batch_size
            iz(:,im) = iz(:,im)+b;
        end
        z{in} = iz;
        if in==nlayer
            exa=exp(iz);
            stan=sum(exa);
            for m = 1:mini_batch_size
                a{in}(:,m)=exa(:,m)/stan(1,m);
            end
        else
            a{in} = sigmoid(iz);
        end
    end

    delta = a{nlayer}-output;
    nabla_bias{nlayer} = mean(delta,2);
    nabla_weight{nlayer} = (delta*(a{nlayer-1})')/mini_batch_size;

    if nlayer>=3
        for in = nlayer-1:-1:2

            delta = weight{in+1}'*delta.*sigmoid_prime(z{in});
            nabla_bias{in} = mean(delta,2);
            nabla_weight{in} = (delta*a{in-1}')/mini_batch_size;
        end
    end

    for in = 2:nlayer
        weight{in} = weight{in}-zeta*nabla_weight{in};
        bias{in} = bias{in}-zeta*nabla_bias{in};
    end
    end
    at{1}=test_inputs;
    for in = 2:nlayer
        wt = weight{in};
        bt = bias{in};
        ixt = at{in-1};
        izt = wt*ixt;
        for im = 1:ntest
            izt(:,im) = izt(:,im)+bt;
        end
        if in==nlayer
            zt{in} = izt;
            exat=exp(izt);
            stant=sum(exat);
            for im = 1:ntest
                at{in}(:,im)=exat(:,im)/stant(1,im);
            end
        else
            zt{in} = izt;
            at{in} = sigmoid(izt);
        end
    end
    for m=1:ntest
        [dx,wz]=max(at{nlayer}(:,m));
        [dx2,wz2]=max(test_bq(:,m));
        if wz==wz2
            s=s+1;
        end
    end
    accuracy(ip+1,1)=s/ntest;
    CrossEntropy(1,ip)=sum(sum(-output.*(log(real(a{2})))))/20;   
    if (accuracy(ip+1,1)==1)&&(ip<max_accu_epoch)
        max_accu_epoch=ip;
    end
    fprintf('%d  %.2f %.2f\n',ip,accuracy(ip+1,1)*100,CrossEntropy(1,ip))
    % 重置计数器
    s=0;
end
weight_final=weight{2};
disp(max_accu_epoch)
subplot(1,2,1)
plot(time,accuracy,'k','linewidth',3);
axis([0 max_epochs,0,1]);
axis([0 max_epochs,0,1]);
xlabel('迭代次数');
ylabel('识别精度');
subplot(1,2,2)
plot(1:max_epochs,CrossEntropy,'k','linewidth',3);
xlabel('迭代次数');
ylabel('交叉熵损失');



