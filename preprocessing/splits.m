
cd '/home/gvarol/datasets/UCF101/splits';

for sp = 1:3
    
    system(['while read p; do mkdir -p split' num2str(sp) '/train/$p; done < ../annot/class_names.txt']);
    system(['while read p; do mkdir -p split' num2str(sp) '/test/$p; done < ../annot/class_names.txt']);
    
    p_train = textread(['../annot/ucfTrainTestlist/trainlist0' num2str(sp) '_1.txt'], '%s');
    for i=1:length(p_train)
        disp([num2str(i) '/' num2str(length(p_train)) ' done.']);
        system(['touch split' num2str(sp) '/train/' p_train{i}]);
    end
    
    p_test = textread(['../annot/ucfTrainTestlist/testlist0' num2str(sp) '_1.txt'], '%s');
    for i=1:length(p_test)
        system(['touch split' num2str(sp) '/test/' p_test{i}]);
    end
    
end

