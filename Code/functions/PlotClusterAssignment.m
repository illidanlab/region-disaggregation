function [] = PlotClusterAssignment(latlon,idx,num_cluster,filename)
figure;
latlim = [min(latlon(:,1)) max(latlon(:,1))];
lonlim = [min(latlon(:,2)) max(latlon(:,2))];
ax = usamap(latlim,lonlim);
set(ax,'visible','off')
latlim = getm(ax,'MapLatLimit');
lonlim = getm(ax,'MapLonLimit');
states = shaperead('usastatehi','UseGeoCoords',true,'BoundingBox',[lonlim',latlim']);
geoshow(ax,states,'FaceColor',[0.9 1 1]);set(gcf,'color','white'); 
set(gcf,'color','white');

sy_color = jet(num_cluster);
for i = 1:num_cluster
    geoshow(latlon(idx==i,1),latlon(idx==i,2),'DisplayType','point','Marker','.'...
        ,'MarkerSize',10,'MarkerEdgeColor',sy_color(i,:))% geoshow(lat,lon)
end
axis off;
% framem off; 
gridm off; mlabel off; plabel off
if ~isempty(filename);
    saveas(gcf,filename,'jpeg');
end

