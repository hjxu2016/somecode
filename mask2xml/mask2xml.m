% example
% MaskPath = 'E:\hjxu-code\xmlAnnotations\9044mask.tif';
% SavePath_xml = 'E:\matlab-code\mat2xml\900.xml';
% Multiple = 16;% Multiple is upsamples
% mask2xml(MaskPath,SavePath_xml,Multiple);
function mask2xml(MaskPath,SavePath_xml,Multiple)
img = imread(MaskPath);
mask1 = im2bw(img,0.9);
[m,n] = find(mask1==1);
[a,~] = size(m);
% [bw_number,~,n1] = bwboundaries(img);

nn = 0;
dataname = [SavePath_xml];
fid = fopen([dataname],'wt');

fprintf(fid, '%s\n', '<Annotations MicronsPerPixel="0.251400">');
fprintf(fid, '%s\n', '  <Annotation Id="1" Name="" ReadOnly="0" NameReadOnly="0" LineColorReadOnly="0" Incremental="0" Type="4" LineColor="65280" Visible="1" Selected="1" MarkupImagePath="" MacroName="">');
fprintf(fid, '%s\n', '    <Attributes>');
fprintf(fid, '%s\n', '      <Attribute Name="Description" Id="0" Value=""/>');
fprintf(fid, '%s\n', '    </Attributes>');
fprintf(fid, '%s\n', '    <Regions>');
fprintf(fid, '%s\n', '      <RegionAttributeHeaders>');
fprintf(fid, '%s\n', '        <AttributeHeader Id="9999" Name="Region" ColumnWidth="-1"/>');
fprintf(fid, '%s\n', '        <AttributeHeader Id="9997" Name="Length" ColumnWidth="-1"/>');
fprintf(fid, '%s\n', '        <AttributeHeader Id="9996" Name="Area" ColumnWidth="-1"/>');
fprintf(fid, '%s\n', '        <AttributeHeader Id="9998" Name="Text" ColumnWidth="-1"/>');
fprintf(fid, '%s\n', '        <AttributeHeader Id="1" Name="Description" ColumnWidth="-1"/>');
fprintf(fid, '%s\n', '      </RegionAttributeHeaders>');
for dith  = 1:a
nn = nn+1;

Region.Id = nn;
Region.Type = 1;
Region.Zoom = 0.020765;
Region.Selected = 0;
Region.ImageLocation = [];
Region.ImageFocus = 0;

Region.Text = [];
Region.NegativeROA = 0; 
Region.InputRegionId = 0;
Region.Analyze = 1;
Region.DisplayId = nn;
fprintf(fid, '%s%d%s%d%s%f%s%d%s%d%s%d%s%f%s%f%s%f%s%f%s%d%s%d%s%d%s%d%s%d%s\n' ...
                               , '      <Region Id="',Region.Id, ...
                                       '" Type="',Region.Type, ...
                                       '" Zoom="',Region.Zoom, ...
                                       '" Selected="',Region.Selected, ...
                                       '" ImageLocation="',Region.ImageLocation, ...
                                       '" ImageFocus="',Region.ImageFocus, ...
                                       '" Text="',Region.Text, ...
                                       '" NegativeROA="',Region.NegativeROA, ...
                                       '" InputRegionId="',Region.InputRegionId, ...
                                       '" Analyze="',Region.Analyze, ...
                                       '" DisplayId="',Region.DisplayId,'">');
fprintf(fid, '%s\n', '        <Attributes/>');
fprintf(fid, '%s\n', '        <Vertices>');

fprintf(fid, '%s%f%s%f%s\n', '          <Vertex X="',n(dith)*Multiple,'" Y="',m(dith)*Multiple,'"/>');
fprintf(fid, '%s\n', '        </Vertices>');
fprintf(fid, '%s\n', '      </Region>');
end
fprintf(fid, '%s\n', '    </Regions>');
% fprintf(fid, '%s\n', '    <Plots/>');
fprintf(fid, '%s\n', '  </Annotation>');
fprintf(fid, '%s\n', '</Annotations>');
fclose(fid);
end