function msle = calculateMSLE(y_pred, y_true)
    % y_true: 真实值矩阵
    % y_pred: 预测值矩阵
    % 检查 y_true 和 y_pred 中是否有负值

    
    % 逐元素计算 MSLE
    msle = sqrt(sum((y_true-y_pred).^2))/sqrt(sum(y_true.^2));  % 按行和列求平均
end
