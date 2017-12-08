clear;
addpath('/home/hjxu/MatlabProjects/openslide-matlab-master/');
addpath('/home/hjxu/lb_code/BOPointInPolygon/')
ImgPath='/media/hjxu/LENOVO/data/data_breast_train/';
xmlPath='/media/hjxu/LENOVO/data/data_breast_train_xml/';
xmlFile=dir([xmlPath,'*.xml']);
for k=1:length(xmlFile)
    oneXml=[xmlPath,xmlFile(k).name];
    c=parseCamelyon16XmlAnnotations(oneXml);
    WSI=[ImgPath,xmlFile(k).name(1:end-4),'.svs'];
    slide=openslide_open(WSI);
    LL=size(c);
    LL=LL(1,1);
    for kk=1:length(LL)
        cell=c{kk,1};
        xmin=min(cell(:,1));
        ymin=min(cell(:,2));
        xmax=max(cell(:,1));
        ymax=max(cell(:,2));
        Img=openslide_read_region(slide,xmin,ymin,xmax-xmin,ymax-ymin,'level',0);
        I=Img(:,:,2:4);
%         figure(2);
%         imshow(I,[]);
        [w,h,~]=size(I);
        height = 256;
        width = 256;
        max_row = floor(h / height);
        max_col = floor(w/ width);
        Lastheight = (h - max_row * height) ;
        Lastwidth = (w - max_col * width);
        filetypeJpg = '.jpg';
        for i=1:max_row
            for j=1:max_col
                pointx=xmin+ j * width+128;
                pointy=ymin+ i * height+128;
                P=c{1,1}(1,:);
                PP = [cell;P];
                point=[pointx,pointy];
                stage = BOPointInPolygon(PP,point);
                disp(['Stage: ' stage]);
                if (stage=='i')
                    patch=openslide_read_region(slide,xmin+ j * width,ymin+ i * height,256,256,'level',0);
                    II=patch(:,:,2:4);
                    JPath=['/home/hjxu/breast_data_256/Write_256_1/',xmlFile(k).name(1:end-4),'/'];
                    if ~exist(JPath)
                        mkdir(JPath);   
                    end
                    JpgPath=[JPath,xmlFile(k).name(1:end-4),'_',num2str(j),'_',num2str(i),filetypeJpg];
                    imwrite(II,JpgPath);
                end
            end
        end
    end
end