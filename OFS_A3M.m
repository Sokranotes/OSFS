function [ selectedFeatures,time ] = OFS_A3M( X, Y)
%  Peng Zhou;Xuegang Hu;Peipei Li
% A New Online Feature Selection Method Using Neighborhood Rough Set
% 2017 IEEE International Conference on Big Knowledge (ICBK) 2017
%
% Output:  selectedFeatures  the index of selected features
%          time   running time
% Input:  X     data samples vector
%         Y     label vector
%


start=tic;
% X中属性个数存到p中，列数
[~,p]=size(X);

% 1*p全零矩阵
mode=zeros(1,p);
dep_Mean=0;
dep_Set=0;
depArray=zeros(1,p);


for i=1:p
     % X中的第i个属性，第i列
     col=X(:,i);
     dep_single=dep_an(col,Y);
     % 第1行第i列
     depArray(1,i)=dep_single;

    if dep_single>dep_Mean
            % 第一行，第i列为1，把当前特征加入到cols中，列对应特征
            mode(1,i)=1;
            % 取出所有选中的特征的值，相当于data，但是只有被选择的特征的值出现
            cols=X(:,mode==1);
            % 计算候选特征子集的依赖度
            dep_New=dep_an(cols,Y);
            if dep_New>dep_Set
                   dep_Set=dep_New;
                   dep_Mean=dep_Set/sum(mode);
            elseif dep_New==dep_Set
                   % 找到non_significance的特征的下标
                   [index_del] = non_signf(X,Y,mode,i);
                   mode(1,index_del)=0;

                   dep_Mean=dep_Set/sum(mode);
           else
                    mode(1,i)=0;
           end

     end

end

selectedFeatures=find(mode==1);

time=toc(start);
end

function [index_del] = non_signf(X,Y,mode,i)

B=zeros(1,length(mode));
R=mode;
T=mode;
T(1,i)=0;

indexs=find(T==1);
Num=length(indexs);
A=randperm(Num);

for i=1:Num
    rnd=A(i);
    ind=indexs(rnd);
    if sig(X,Y,R,ind)==0
        B(1,ind)=1;
        R(1,ind)=0;
    end
end
index_del=B==1;
end

function [s]= sig(X,Y,mode,f_i)

[d_B]=dep_an(X(:,mode==1),Y);
mode(1,f_i)=0;
[d_F]=dep_an(X(:,mode==1),Y);

s=(d_B-d_F)/d_B;
end


function [ dep ] = dep_an(data,Y)

% 数据维数（样本个数）为n，行数
[n,~]=size(data);
% 标记长度（所有样本个数）
card_U=length(Y);
card_ND=0;
% 计算标准欧几里德距离Standardized Euclidean distance
D = pdist(data,'seuclidean');
% 转换成距离方阵，元素表示i和j之间的标准欧几里得距离
DArray=squareform(D,'tomatrix');
% 遍历每个样本
for i=1:n
     % 第i个样本与其他样本的距离向量
     d=DArray(:,i);
     % 取出第i个label
     class=Y(i,1);
     % d距离向量，Y标记向量，class该样本的标记，n总样本数
     card_ND=card_ND+card(d,Y,class,n);
end
dep=card_ND/card_U;
end


% d距离向量，Y标记向量，class该样本的标记，n总样本数
function [c]=card(sets,Y,label,N)
        % 从小到大排序，D中存排好序之后的距离，I中存排好序之后原先的index
        [D,I]=sort(sets);

        % 最小距离
        min_d=D(2,1);
        % 最大距离
        max_d=D(N,1);
        % gap
        mean_d=1.5*(max_d-min_d)/(N-2);

        % card值并不包括其本身计算
        cNum=0;
        % gap邻域中包含其本身
        cTotal=1;
        % 最近的样本的下标
        ind2=I(2,1);
        if Y(ind2,1)==label
               cNum=cNum+1;
        end

         for j=3:N
             if D(j,1)-D(j-1,1)<mean_d
                 % 在邻域内的样本的下标
                 ind=I(j,1);
                 % gap邻域中的样本数+1
                 cTotal=cTotal+1;
                 if Y(ind,1)==label
                     % label一致，则card值加一
                     cNum=cNum+1;
                 end
             else
                  break;
             end
         end

         % 在邻域中相同标签样本个数除以邻域大小
         c=cNum/cTotal;

end
