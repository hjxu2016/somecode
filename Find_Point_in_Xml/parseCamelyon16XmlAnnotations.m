function c = parseCamelyon16XmlAnnotations(xmlPath)
xmlDocument = xmlread(xmlPath);
import javax.xml.xpath.*;
factory = XPathFactory.newInstance();
xpath = factory.newXPath();
% XPath expressions used to parse the XML file
annotationExpr = xpath.compile('//Annotation');
PartOfGroup=xpath.compile('//Region/@Id');
xExpr = xpath.compile('.//Vertex/@X');
yExpr = xpath.compile('.//Vertex/@Y');
annotations = annotationExpr.evaluate(xmlDocument, XPathConstants.NODESET);
for i_annotations = annotations.getLength():-1:1;
    
    current = annotations.item(i_annotations-1);
    
    p=PartOfGroup.evaluate(current, XPathConstants.NODESET);
    x = xExpr.evaluate(current, XPathConstants.NODESET);
    y = yExpr.evaluate(current, XPathConstants.NODESET);
    
    currentCoordinates = zeros(x.getLength(), 2);
    %Part=zeros(p.getLength(), 1);
    Part(1,i_annotations)={p.item(i_annotations-1).getValue};
    %   currentCoordinates(i_annotations, 2) = ...
    %    % convert the coordinates to a MATLAB array
    for i_coordinate = 1:x.getLength()
        currentCoordinates(i_coordinate, 1) = ...
            floor(str2double(x.item(i_coordinate-1).getValue));
        currentCoordinates(i_coordinate, 2) = ...
            floor(str2double(y.item(i_coordinate-1).getValue));
    end
    coordinates{i_annotations,1} = currentCoordinates;
end
Part = Part';
c=[coordinates Part];

end
