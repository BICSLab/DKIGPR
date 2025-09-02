% clear all;
% clc;
y=1;
load train.mat
% load layer3.mat
% load encoder3.mat
% load decoder3.mat

% train = train(1:2250,:);
% load traintotal.mat
% train = train(1:2250,1:3);
train = train(1:end,1:3);
train_min = min(train);
train_max = max(train);
train(:,1) = (train(:,1)-train_min(:,1))/(train_max(:,1)-train_min(:,1));
train(:,2) = (train(:,2)-train_min(:,2))/(train_max(:,2)-train_min(:,2));
train(:,3) = (train(:,3)-train_min(:,3))/(train_max(:,3)-train_min(:,3));
% train(:,4) = (train(:,4)-train_min(:,4))/(train_max(:,4)-train_min(:,4));
% train(:,5) = (train(:,5)-train_min(:,5))/(train_max(:,5)-train_min(:,5));

train = train(1:end,:);

[numPoints, numFeatures] = size(train(:,2:end));
data = train(:,1:end)';
u = train(:,1)';

layersEncoder = [
    % 特征输入层
    featureInputLayer(numFeatures);
    % 全连接层-1
    fullyConnectedLayer(8)
    eluLayer
    % 全连接层-2
    fullyConnectedLayer(8)
   


    % fullyConnectedLayer(80)
    % eluLayer
];

layersfully = [    featureInputLayer(9);

    fullyConnectedLayer(8,"BiasLearnRateFactor",0)
];

layersDecoder = [
    featureInputLayer(9)
    
    fullyConnectedLayer(8)
    eluLayer
    % 回归输出层
    % fullyConnectedLayer(20)
    % eluLayer
    fullyConnectedLayer(numFeatures)];

dlnetEncoder = dlnetwork(layersEncoder);
dlnetDecoder = dlnetwork(layersDecoder);
dlnetlayer = dlnetwork(layersfully);
% analyzeNetwork(dlnetEncoder)
% analyzeNetwork(dlnetDecoder)rux;sk,lu.;i/o5poooodo/o/co/ordoi.;e5x55sl.i;5si;s5s5,lsiis5ssis.i5;.,lykr,uli.xr,lu.ixeii.;i.i.;5;5oos.i;i;ei.;o/exxeexe5oo;.i;e.e.e.e.e.e.ei;oeoeo/eo/oe;o;xoxoeo/exoxoeox;xo/oxoxo/o/o/o/cdo/o/exxe/e/eo/o/eoxoexe;exexoxoxe
min1 = 0.005;
options.batchSize = 100;
options.maxEpochs = 500;
options.learnRate = 0.001;
options.gradDecay = 0.75; 
options.sqGradDecay = 0.99;
numIterations = floor(numPoints./options.batchSize);
figure
lineLossTrain = animatedline('Color', [0.85 0.325 0.098]);
ylim([0 inf])
xlabel("Iteration")
ylabel("Loss")
grid on
box on

averageGradEn = []; % 过去梯度的指数加权平均值
averageSqGradEn = []; % 过去梯度的平方的指数加权平均值

% 解码器
averageGradDe = []; % 过去梯度的指数加权平均值
averageSqGradDe = []; % 过去梯度的平方的指数加权平均值

averageGradL = []; % 过去梯度的指数加权平均值
averageSqGradL = []; % 过去梯度的平方的指数加权平均值

% 训练
globalIterations  = 0;
start = tic;
for epoch = 1:options.maxEpochs
    % Shuffle data
    % shuffleIndex = randperm(numPoints);
    shuffleIndex = (1:numPoints);
    shuffleData = data(2:end, shuffleIndex);
    shuffleu = u(:,shuffleIndex);
    for i = 1:numIterations
        globalIterations  = globalIterations + 1;
        idx = (i-1)*options.batchSize+1:i*options.batchSize;
        batchData = (dlarray(shuffleData(:, idx), 'CB'));
        batchu = (shuffleu(:,idx));
        [gradientsEn, gradientsDe,gradientsL, loss] = dlfeval(@AEmodelGradients, dlnetEncoder, dlnetDecoder,dlnetlayer, batchData,batchu);
        % if(double(gather(extractdata(loss)))<min1)
        % 
        %     min1 = double(gather(extractdata(loss)));
        %     disp(min1);
        %     save('encoder.mat','dlnetEncoder');
        %     save('decoder.mat','dlnetDecoder');
        % end
        % 使用 Adam 优化器来更新网络参数.
        if(i<400)
        [dlnetEncoder,averageGradEn, averageSqGradEn] = adamupdate...
            (dlnetEncoder,gradientsEn,averageGradEn,averageSqGradEn, globalIterations,...
            options.learnRate, options.gradDecay, options.sqGradDecay);

        [dlnetDecoder,averageGradDe, averageSqGradDe] = adamupdate...
            (dlnetDecoder, gradientsDe, averageGradDe, averageSqGradDe, globalIterations,...
            options.learnRate, options.gradDecay, options.sqGradDecay);

        [dlnetlayer,averageGradL, averageSqGradL] = adamupdate...
            (dlnetlayer, gradientsL, averageGradL, averageSqGradL, globalIterations,...
            options.learnRate, options.gradDecay, options.sqGradDecay);
        else
                    [dlnetEncoder,averageGradEn, averageSqGradEn] = adamupdate...
            (dlnetEncoder,gradientsEn,averageGradEn,averageSqGradEn, globalIterations,...
            options.learnRate, options.gradDecay, options.sqGradDecay);

                            [dlnetDecoder,averageGradDe, averageSqGradDe] = adamupdate...
            (dlnetDecoder, gradientsDe, averageGradDe, averageSqGradDe, globalIterations,...
            options.learnRate, options.gradDecay, options.sqGradDecay);
        end

        % 可视化训练过程
        D = duration(0,0,toc(start),'Format','hh:mm:ss');
        addpoints(lineLossTrain,globalIterations,double(gather(extractdata(loss))))
        title("Epoch: " + epoch + ", Elapsed: " + string(D))
        drawnow
    end
end

% globalIterations  = 0;
% start = tic;
% for epoch = 1:options.maxEpochs
%     % Shuffle data
%     % shuffleIndex = randperm(numPoints);
%     shuffleIndex = (1:numPoints);
%     shuffleData = data(:, shuffleIndex);
%     shuffleu = u(:,shuffleIndex);
%     for i = 1:numIterations
%         globalIterations  = globalIterations + 1;
%         idx = (i-1)*options.batchSize+1:i*options.batchSize;
%         batchData = gpuArray(dlarray(shuffleData(:, idx), 'CB'));
%         batchu = gpuArray(shuffleu(:,idx));
%         [gradientsEn, gradientsDe, loss,AB] = dlfeval(@AEmodelGradients1, dlnetEncoder, dlnetDecoder, batchData,batchu);
% 
%         % 使用 Adam 优化器来更新网络参数.
%         [dlnetEncoder,averageGradEn, averageSqGradEn] = adamupdate...
%             (dlnetEncoder,gradientsEn,averageGradEn,averageSqGradEn, globalIterations,...
%             options.learnRate, options.gradDecay, options.sqGradDecay);
% 
%         [dlnetDecoder,averageGradDe, averageSqGradDe] = adamupdate...
%             (dlnetDecoder, gradientsDe, averageGradDe, averageSqGradDe, globalIterations,...
%             options.learnRate, options.gradDecay, options.sqGradDecay);
% 
%         % 可视化训练过程
%         E = duration(0,0,toc(start),'Format','hh:mm:ss');
%         addpoints(lineLossTrain,globalIterations,double(gather(extractdata(loss))))
%         title("Epoch: " + epoch + ", Elapsed: " + string(E))
%         drawnow
%     end
% end
save('encoder4.mat','dlnetEncoder');
save('decoder4.mat','dlnetDecoder');
save('layer4.mat','dlnetlayer');

% function [gradientsEn, gradientsDe, loss,AB] = AEmodelGradients1(dlnetEncoder, dlnetDecoder, batchData,batchu)
%     % 计算编码器的输出
%     encodedData = forward(dlnetEncoder, batchData(1:4,:));
%     % 计算解码器的输出
%     decodedData = forward(dlnetDecoder, [encodedData;dlarray(batchu,'CB')]);
%     tmp_past = [extractdata(encodedData(:,1:end-1));batchu(:,1:end-1)];
%     tmp_now = extractdata(encodedData(:,2:end));
%     AB = tmp_now*tmp_past'*pinv(tmp_past*tmp_past');
%     % g_prd = AB*tmp_past;
%     g_prd = [];
%     x_koop = tmp_past(:,1);
%     for i=1:size(tmp_now,2)
%         x_koop = AB*x_koop;
%         g_prd=[g_prd,x_koop];
%         x_koop = [x_koop;batchu(:,i+1)];
%     end
% 
%     g_prd = dlarray(g_prd,'CB');
%     y_prd = forward(dlnetDecoder, [g_prd;dlarray(batchu(:,2:end),'CB')]);
% 
%     % y_prd = forward(dlnetDecoder, [tmp_now;dlarray(batchu(:,2:end),'CB')]);
% 
%     % % 计算loss
%     loss_linear = mse(g_prd,encodedData(:,2:end));
%     loss_con = mse(decodedData, batchData(1:4,:));
%     loss_prd = mse(y_prd,batchData(1:4,2:end));
% 
%     loss = loss_linear+loss_con+loss_prd;
%     % loss = loss_con;
%     % 计算梯度
%     [gradientsEn, gradientsDe] = dlgradient(loss, dlnetEncoder.Learnables, dlnetDecoder.Learnables);
% end

function [gradientsEn, gradientsDe,gradientsL, loss] = AEmodelGradients(dlnetEncoder, dlnetDecoder,dlnetlayer, batchData,batchu)
    % 计算编码器的输出
    encodedData = forward(dlnetEncoder, batchData(1:end,:));
    % 计算解码器的输出
    decodedData = forward(dlnetDecoder, [encodedData;dlarray(batchu,'CB')]);
    % tmp_past = [extractdata(encodedData(:,1:end-1));batchu(:,1:end-1)];
    % tmp_now = extractdata(encodedData(:,2:end));
    % AB = tmp_now*tmp_past'*pinv(tmp_past*tmp_past');
    % g_prd = AB*tmp_past;
    batchu = dlarray(batchu,'CB');
    x_koop = encodedData(:,1);
    g_prd = x_koop;
    for i=1:size(encodedData,2)-1
        x_koop = [x_koop;batchu(:,i)];
        x_koop = forward(dlnetlayer, x_koop);
        g_prd=[g_prd,x_koop];       
    end
    g_prd = dlarray(g_prd,'CB');
    y_prd = forward(dlnetDecoder, [g_prd;dlarray(batchu(:,1:end),'CB')]);
    % 计算loss
    loss_linear = mse(g_prd,encodedData);
    loss_con = mse(decodedData, batchData(1:end,:));
    loss_prd1 = mse(y_prd(1:end,:),batchData(1:end,:));
    lambdaL1 = 0.001;
    c = dlnetlayer.Learnables{1, 'Value'}{1};
    lossL1 = lambdaL1 * sum(sum(abs(c)));
 
    loss = 0.1*loss_con+loss_prd1+loss_linear;
    % loss = loss_con;
    % 计算梯度
    [gradientsEn, gradientsDe,gradientsL] = dlgradient(loss, dlnetEncoder.Learnables, dlnetDecoder.Learnables,dlnetlayer.Learnables);
end
