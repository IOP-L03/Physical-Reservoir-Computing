clc
clear
close all;
letter=xlsread('letter.xlsx')';
target=diag(ones(20,1));

larger=10;
[row,col]=size(letter);

t=zeros(row,col,larger-1);
for i=1:larger-1
    t=imnoise(letter(1:row,:),'salt & pepper',0.05);
    letter(row*i+1:row*(i+1),:)=t();
end

for i=1:larger-1
    target(:,col*i+1:col*(i+1))=target(:,1:col);
end


state = [0.0, 8.314, 7.603, 16.859,6.917,16.816,14.823,24.789,...
    7.115,16.177,14.101,25.118,15.333,23.401,22.758,32.286];
inputs=zeros(5,20);
for i=1:20*larger
    for j=1:5
       order=letter(i,4*j-3)*8+letter(i,4*j-2)*4+letter(i,4*j-1)*2+letter(i,4*j)+1;
       inputs(j,i) =state(order);
    end
end
inputs_max=max(max(inputs));
inputs_min=min(min(inputs));
inputs=(inputs-inputs_min)/(inputs_max-inputs_min);

row=row*larger;
col=col*larger;

colrank=randperm(col);
new_inputs=inputs(:,colrank);
new_target=target(:,colrank);

train_input=new_inputs(:,1: round(col*0.7));
train_target=new_target(:,1:round(col*0.7));
test_input=new_inputs(:,round(col*0.7)+1:end);
test_target=new_target(:,round(col*0.7)+1:end);
[mtrain,ntrain]=size(train_target);
[mtest,ntest]=size(test_target);

arch = [5,20];
nlayer = length(arch);

mini_batch_size = 100;
max_epochs = 2000;
max_accu_epoch=max_epochs;
zeta =5;
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
    weight{in} = rand(arch(1,in),arch(1,in-1))*2-1;
    bias{in} = rand(arch(1,in),1);
    nabla_weight{in} = rand(arch(1,in),arch(1,in-1));
    nabla_bias{in} = rand(arch(1,in),1);
end
for in = 1:nlayer
    a{in} = zeros(arch(in),mini_batch_size);
    z{in} = zeros(arch(in),mini_batch_size);
    at{in} = zeros(arch(in),ntest);
    zt{in} = zeros(arch(in),ntest);
end

time=0:max_epochs;
accuracy=zeros(max_epochs+1,1);

s=0;

at{1}=test_input;
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
        at{in} = relu(izt);
    end
end
for m=1:ntest
    [dx,wz]=max(at{nlayer}(:,m));
    [dx2,wz2]=max(test_target(:,m));
    if wz==wz2
        s=s+1;
    end
end
accuracy(1,1)=s/ntest;

s=0;
for ip = 1:max_epochs
    
    pos = randi(ntrain+1-mini_batch_size);
    input = train_input(:,pos:pos+mini_batch_size-1);
    output = train_target(:,pos:pos+mini_batch_size-1);
    
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
            a{in} = relu(iz);
        end
    end

    delta = a{nlayer}-output;
    nabla_bias{nlayer} = mean(delta,2);
    nabla_weight{nlayer} = (delta*(a{nlayer-1})')/mini_batch_size;

    if nlayer>=3
        for in = nlayer-1:-1:2
            delta = weight{in+1}'*delta.*relu_prime(z{in});
            nabla_bias{in} = mean(delta,2);
            nabla_weight{in} = (delta*a{in-1}')/mini_batch_size;
        end
    end

    for in = 2:nlayer
        weight{in} = weight{in}-zeta*nabla_weight{in};
        bias{in} = bias{in}-zeta*nabla_bias{in};
    end
    end

    at{1}=test_input;
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
            at{in} = relu(izt);
        end
    end
    for m=1:ntest
        [dx,wz]=max(at{nlayer}(:,m));
        [dx2,wz2]=max(test_target(:,m));
        if wz==wz2
            s=s+1;
        end
        %fprintf(' %d ',wz)
    end
    accuracy(ip+1,1)=s/ntest;
    if (accuracy(ip+1,1)==1)&&(ip<max_accu_epoch)
        max_accu_epoch=ip;
    end
    fprintf('%d  %.2f\n',ip,accuracy(ip+1,1)*100)

    s=0;
end
disp(max_accu_epoch)
disp(accuracy(201,1))

figure,plot(time,accuracy,'k','linewidth',3);
axis([0 max_epochs,0,1]);
xlabel('epoch');
ylabel('acurracy');