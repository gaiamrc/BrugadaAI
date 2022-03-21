function plot1lead(leads,timefct,titles)
    for i = 1:12
        figure
        time = (0:(length(leads(:,i))-1))*timefct;
        
        ymin = -20;
        ymax = 20;
        x1 = 0:1:(numel(time))*timefct;
        y1 = ymin:1:ymax;
        [X11,Y11] = meshgrid(x1,y1);
        [X12,Y12] = meshgrid(y1,x1);
        plot(X11,Y11,'Color',[0.9882    0.7804    0.7804],'Linewidth',0.25)
        hold on
        plot(Y12,X12,'Color',[0.9882    0.7804    0.7804],'Linewidth',0.25)
        hold on
        
        x2 = 0:5:(numel(time))*timefct;
        y2 = ymin:5:ymax;
        [X21,Y21] = meshgrid(x2,y2);
        [X22,Y22] = meshgrid(y2,x2);
        plot(X21,Y21,'Color',[1.0000    0.4902    0.4902],'Linewidth',0.5)
        hold on
        plot(Y22,X22,'Color',[1.0000    0.4902    0.4902],'Linewidth',0.5)
        hold on
    
        
        plot(time,leads(1:length(leads),i),'k')
        title(titles(i))
        axis([0 250 ymin ymax])
        set(gca,'DataAspectRatio',[1 1 1])
    
    end
end