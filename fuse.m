for i=0:40
    img3=double(imread(['results/EDVR_BD/calendar/' num2str(i,'%08d') '.png']));
    %img2=double(imread(['Stage2/calendar/' num2str(i,'%08d') '.png']));
    %img3=double(imread(['EDVR_BD/calendar/calendar_' num2str(i+1,'%03d') '.png']));
    img4=double(imread(['results/IRN_3stages/calendar/' num2str(i,'%08d') '.png']));
    %img4=double(imread(['IRN_3stages/calendar/calendar_' num2str(i+1,'%03d') '.png']));
    %img5=double(imread(['Stage5/calendar/' num2str(i,'%08d') '.png']));
    img=uint8((0.5*img3+0.5*img4));%/2
    img4=uint8(img4);
    if ~exist('results/IRN_3stages_parallel/calendar/','dir')
        mkdir('results/IRN_3stages_parallel/calendar/');
    end
    imwrite(img,['results/IRN_3stages_parallel/calendar/' num2str(i,'%08d') '.png'])
end

for i=0:33
    img3=double(imread(['results/EDVR_BD/city/' num2str(i,'%08d') '.png']));
    %img2=double(imread(['Stage2/city/' num2str(i,'%08d') '.png']));
    %img3=double(imread(['RSDN/city/city_' num2str(i+1,'%03d') '.png']));
    img4=double(imread(['results/IRN_3stages/city/' num2str(i,'%08d') '.png']));
    %img4=double(imread(['IRN_3stages/city/city_' num2str(i+1,'%03d') '.png']));
    %img5=double(imread(['Stage5/city/' num2str(i,'%08d') '.png']));
    img=uint8((0.5*img3+0.5*img4));%/2
    img4=uint8(img4);
    if ~exist('results/IRN_3stages_parallel/city/','dir')
        mkdir('results/IRN_3stages_parallel/city/');
    end
    imwrite(img,['results/IRN_3stages_parallel/city/' num2str(i,'%08d') '.png'])
end

for i=0:48
    img3=double(imread(['results/EDVR_BD/foliage/' num2str(i,'%08d') '.png']));
    %img2=double(imread(['Stage2/foliage/' num2str(i,'%08d') '.png']));
    %img3=double(imread(['RSDN/foliage/foliage_' num2str(i+1,'%03d') '.png']));
    img4=double(imread(['results/IRN_3stages/foliage/' num2str(i,'%08d') '.png']));
    %img4=double(imread(['IRN_3stages/foliage/foliage_' num2str(i+1,'%03d') '.png']));
    %img5=double(imread(['Stage5/foliage/' num2str(i,'%08d') '.png']));
    img=uint8((0.5*img3+0.5*img4));%/2
    img4=uint8(img4);
    if ~exist('results/IRN_3stages_parallel/foliage/','dir')
        mkdir('results/IRN_3stages_parallel/foliage/');
    end
    imwrite(img,['results/IRN_3stages_parallel/foliage/' num2str(i,'%08d') '.png'])
end

for i=0:46
    img3=double(imread(['results/EDVR_BD/walk/' num2str(i,'%08d') '.png']));
    %img2=double(imread(['Stage2/walk/' num2str(i,'%08d') '.png']));
    %img3=double(imread(['RSDN/walk/walk_' num2str(i+1,'%03d') '.png']));
    img4=double(imread(['results/IRN_3stages/walk/' num2str(i,'%08d') '.png']));
    %img4=double(imread(['IRN_3stages/walk/walk_' num2str(i+1,'%03d') '.png']));
    %img5=double(imread(['Stage5/walk/' num2str(i,'%08d') '.png']));
    img=uint8((0.5*img3+0.5*img4));%/2
    img4=uint8(img4);
    if ~exist('results/IRN_3stages_parallel/walk/','dir')
        mkdir('results/IRN_3stages_parallel/walk/');
    end
    imwrite(img,['results/IRN_3stages_parallel/walk/' num2str(i,'%08d') '.png'])
end
